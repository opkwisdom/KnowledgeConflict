import torch.nn as nn
import torch
import logging
import re
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from typing import Union
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM

from utils import compute_metrics, parse_reference_answer, load_h5_scores, compute_ndcg
from models import load_model, CAFormerGGClassifier, RankwiseGuideLoss, PairwiseRankGuideLoss

logger = logging.getLogger(__name__)


class GenLossClfLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig,
                 llm: AutoModelForCausalLM, llm_tokenizer: AutoTokenizer, caformer_clf: CAFormerGGClassifier):
        super().__init__()
        self.save_hyperparameters(ignore=["llm", "caformer_clf"])

        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.caformer_clf = caformer_clf
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.top_r = getattr(cfg, "topk_per_query", 10)
        self.n_iter = getattr(cfg, "n_iter", 1)
        self.gamma = getattr(cfg, "gamma", 1.0)
        self.alpha = getattr(cfg, "alpha", 0.5) 

        self.score_transform = getattr(cfg, "score_transform", None)
        self.scaling_factor = getattr(cfg, "scaling_factor", 1.0)
        self.T = getattr(cfg, "T", 1.0)
        # self.pointwise_guide_loss_fn = nn.MSELoss()
        self.pointwise_guide_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.rankwise_guide_loss_fn = RankwiseGuideLoss()
        # self.rankwise_guide_loss_fn = PairwiseRankGuideLoss()
        self.gen_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.h5_scores = load_h5_scores(self.cfg.precompute_table_path)
        self.val_preds = []
        self.val_labels = []
        self.val_ndcg_scores = []
        self.prepare_modules()

    def prepare_modules(self):
        # Freeze the pretrained LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        
        for param in self.caformer_clf.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()  # Ensure the base model is always in eval mode

    def forward(self, batch):
        """
        Forward pass to generate LLM hidden states for both documents and CA-Former input.
        Returns:
            llm_repr: Tensor of shape (B*K, L, S, D_llm) - LLM hidden states for CA-Former input
        """
        # Generate LLM hidden states for CA-Former input
        with torch.no_grad():
            llm_outputs = self.llm.model(
                input_ids=batch["source_input_ids"],
                attention_mask=batch["source_attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            llm_repr = torch.stack(llm_outputs.hidden_states[-12:]).permute(1, 0, 2, 3)   # (B*K, L, S, D_llm)
        return llm_repr
    
    @torch.inference_mode()
    def generate_answer(self, batch, top_r_indices):
        """
        Compute the generation loss conditioned on the interleaving document inputs.
        This is computed using only re-ranked docs(top_r_indices).
        Args:
            batch: The input batch containing document, question, and answer IDs.
            top_r_indices: Tensor of shape (B, R)
        Returns:
            batch_preds_text: List[str] of length B
            batch_refs_text: List[str] of length B
        """
        # Meta-infos
        doclen_list = batch["doclen_list"]  # (B, K)
        B, K = doclen_list.shape
        
        # Reshape inputs
        doc_input_ids = batch["doc_input_ids"].reshape(B, K, -1)
        question_ids = batch["question_ids"]
        answer_ids = batch["answer_ids"]
        a_len_list = batch["a_len"]

        input_embeds_list = []
        labels_list = []

        for i in range(B):
            sample_top_r_indices = top_r_indices[i]
            sample_doc_input_ids = doc_input_ids[i]
            sample_doclen_list = doclen_list[i]
            sample_question_ids = question_ids[i]

            sample_input_ids = []
            for idx in sample_top_r_indices:
                doclen = sample_doclen_list[idx]
                doc_ids = sample_doc_input_ids[idx, :doclen]
                sample_input_ids.append(doc_ids)
            
            sample_input_ids.append(sample_question_ids)
            sample_input_ids = torch.cat(sample_input_ids) # (L,)

            # Embedding
            sample_input_embeds = self.llm.get_input_embeddings()(sample_input_ids)   # (L, D)
            input_embeds_list.append(sample_input_embeds)
        
        max_len = max(input_embeds.shape[0] for input_embeds in input_embeds_list)
        padded_input_embeds = pad_sequence(input_embeds_list, batch_first=True, padding_value=0.0, padding_side='left')

        seq_lens = torch.tensor([input_embeds.shape[0] for input_embeds in input_embeds_list], device=padded_input_embeds.device)
        mask_range = torch.arange(max_len, device=padded_input_embeds.device).unsqueeze(0)
        padded_attention_mask = (mask_range < seq_lens.unsqueeze(1)).long()

        generated_ids = self.llm.generate(
            inputs_embeds=padded_input_embeds,
            attention_mask=padded_attention_mask,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )

        batch_preds_text = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        batch_refs_text = []
        for i in range(B):
            a_len = a_len_list[i].item()
            ref_ids = answer_ids[i][:a_len]
            ref_text = self.llm_tokenizer.decode(ref_ids, skip_special_tokens=True)
            batch_refs_text.append(ref_text)

        return batch_preds_text, batch_refs_text
    
    def transform_scores_oracle(self, scores_oracle):
        if self.cfg.score_transform is None:
            return self.scaling_factor * scores_oracle
        elif self.cfg.score_transform == "sigmoid":
            return torch.sigmoid(scores_oracle / self.T)
        elif self.cfg.score_transform == "tanh":
            return torch.tanh(scores_oracle / self.T)
        else:
            raise ValueError(f"Unknown score transform: {self.cfg.score_transform}")

    def reranking_documents(self, batch, scores_hat, scores_oracle):
        """
        Re-rank documents based on scores, return Top-R documents
        , where R << K
        """
        B, K = batch["doclen_list"].shape
        scores_hat_reshaped = scores_hat.reshape(B, K)
        scores_oracle_reshaped = scores_oracle.reshape(B, K)
        # lmbda = self.get_reranking_scheduled_weight()
        
        # scores = (1 - lmbda) * scores_oracle_reshaped + lmbda * scores_hat_reshaped
        topk_student_indices = scores_hat_reshaped.sort(dim=-1, descending=True)[1]

        scores_hat_np = scores_hat_reshaped.cpu().float().numpy()
        scores_oracle_np = scores_oracle_reshaped.cpu().float().numpy()
        
        batch_ndcg = []
        for i in range(B):
            sample_ndcg = compute_ndcg(scores_hat_np[i], scores_oracle_np[i])
            batch_ndcg.append(sample_ndcg)

        topr_indices = topk_student_indices[:, :self.top_r]
        return topr_indices, batch_ndcg

    def training_step(self, batch, batch_idx):
        llm_repr = self.forward(batch)
        scores_hat, _ = self.caformer_clf(llm_repr, batch["source_attention_mask"],
                                                            batch["roberta_question_ids"], batch["roberta_question_mask"])
        del llm_repr

        raw_scores_oracle = batch["scores_oracle"].reshape(-1, 1)   # (B*K, 1)
        scores_oracle = raw_scores_oracle.to(scores_hat.device)
        scores_oracle = self.transform_scores_oracle(scores_oracle)

        # Guide loss (Pointwise + Rankwise)
        scores_hat = scores_hat.view(-1)
        scores_oracle = scores_oracle.view(-1)
        raw_pointwise_guide_loss = self.pointwise_guide_loss_fn(scores_hat, scores_oracle)
        B, K = batch["doclen_list"].shape
        raw_rankwise_guide_loss = self.rankwise_guide_loss_fn(scores_hat.reshape(B, K), scores_oracle.reshape(B, K))
        
        pointwise_guide_loss = (1 - self.alpha) * raw_pointwise_guide_loss
        rankwise_guide_loss = self.alpha * raw_rankwise_guide_loss
        loss = pointwise_guide_loss + rankwise_guide_loss
        
        self.log("train/pointwise_guide_loss", raw_pointwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/rankwise_guide_loss", raw_rankwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        llm_repr = self.forward(batch)
        scores_hat, _ = self.caformer_clf(llm_repr, batch["source_attention_mask"],
                                                            batch["roberta_question_ids"], batch["roberta_question_mask"])
        del llm_repr

        # Compute the gradient of the oracle loss and target score
        raw_scores_oracle = batch["scores_oracle"].reshape(-1, 1)   # (B*K, 1)
        scores_oracle = raw_scores_oracle.to(scores_hat.device)
        scores_oracle = self.transform_scores_oracle(scores_oracle)

        # Guide loss (Pointwise + Rankwise)
        scores_hat = scores_hat.view(-1)
        scores_oracle = scores_oracle.view(-1)
        raw_pointwise_guide_loss = self.pointwise_guide_loss_fn(scores_hat, scores_oracle)
        B, K = batch["doclen_list"].shape
        raw_rankwise_guide_loss = self.rankwise_guide_loss_fn(scores_hat.reshape(B, K), scores_oracle.reshape(B, K))
        
        pointwise_guide_loss = (1 - self.alpha) * raw_pointwise_guide_loss
        rankwise_guide_loss = self.alpha * raw_rankwise_guide_loss
        loss = pointwise_guide_loss + rankwise_guide_loss
        
        # TODO: Re-ranking, select Top-R
        top_r_indices, batch_ndcg = self.reranking_documents(batch, scores_hat, raw_scores_oracle)
        self.val_ndcg_scores.extend(batch_ndcg)

        # Generation quality evaluation
        batch_preds, batch_labels = self.generate_answer(batch, top_r_indices)
        self.val_preds.extend(batch_preds)
        self.val_labels.extend(batch_labels)

        # self.log("valid/oracle_score", scores_oracle, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/pointwise_guide_loss", raw_pointwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid/rankwise_guide_loss", raw_rankwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid/loss", loss, on_step=False, on_epoch=True, sync_dist=True)


    def parse_final_answer(self, text: str) -> str:
        match = re.search(r'Final Answer:\s*(.*)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def on_validation_epoch_end(self):
        total_em = 0.0
        total_f1 = 0.0
        total_count = 0
        
        # Compute EM, F1 based on val_preds and val_labels
        for pred, answer in zip(self.val_preds, self.val_labels):
            metrics = compute_metrics(pred, answer)

            total_em += float(metrics.soft_em)
            total_f1 += float(metrics.f1)
            total_count += 1

        avg_em = total_em / total_count if total_count > 0 else 0.0
        avg_f1 = total_f1 / total_count if total_count > 0 else 0.0
        
        # Compute average NDCG
        np_ndcg = np.array(self.val_ndcg_scores)
        avg_ndcg = np.mean(np_ndcg, axis=0)

        self.log("valid/EM", avg_em, sync_dist=True)
        self.log("valid/F1", avg_f1, sync_dist=True)
        self.log("valid/NDCG@1", avg_ndcg[0], sync_dist=True)
        self.log("valid/NDCG@3", avg_ndcg[1], sync_dist=True)
        self.log("valid/NDCG@5", avg_ndcg[2], sync_dist=True)
        self.log("valid/NDCG@10", avg_ndcg[3], sync_dist=True)

        self.val_preds.clear()
        self.val_labels.clear()
        self.val_ndcg_scores.clear()
        
    def on_save_checkpoint(self, checkpoint):
        # save only CAFormer classifier weights
        state_dict = checkpoint["state_dict"]
        caformer_clf_state_dict = {
            k: v for k, v in state_dict.items() if "caformer_clf" in k
        }
        checkpoint["state_dict"] = caformer_clf_state_dict

    def configure_optimizers(self):
        # trainable_params = filter(lambda p: p.requires_grad, self.caformer_clf.parameters())
        caformer_params = [
            p for p in self.caformer_clf.caformer.parameters() if p.requires_grad
        ]
    
        classifier_params = [
            p for p in self.caformer_clf.classifier.parameters() if p.requires_grad
        ]

        # Separate parameter groups, Classifier is random initialized
        optimizer_grouped_parameters = [
            {"params": caformer_params, "lr": self.learning_rate},
            {"params": classifier_params, "lr": 1e-3}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, fused=True)

        # optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, fused=True)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }