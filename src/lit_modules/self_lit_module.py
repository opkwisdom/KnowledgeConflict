import torch.nn as nn
import torch
import logging
import re
import numpy as np
from src.utils import template
from torch.nn.utils.rnn import pad_sequence
from typing import Union
from pytorch_lightning import LightningModule
from scipy.stats import spearmanr
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM

from utils import compute_metrics, parse_reference_answer, load_h5_scores, compute_ndcg, compute_recall
from models import SelfGenLossModel, ListwiseGuideLoss

logger = logging.getLogger(__name__)


class GenLossSelfLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig,
                 llm: AutoModelForCausalLM, llm_tokenizer: AutoTokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=["llm"])

        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.top_r = getattr(cfg, "topk_per_query", 10)
        self.alpha = getattr(cfg, "alpha", 0.5)

        # Soft prompt model
        n_soft_tokens = getattr(cfg, "query_length", 32)
        hidden_dim = self.llm.config.hidden_size
        self.soft_model = SelfGenLossModel(llm, n_soft_tokens, hidden_dim)

        # self.pointwise_guide_loss_fn = nn.MSELoss()
        self.pointwise_guide_loss_fn = nn.SmoothL1Loss(reduction='mean')
        # self.rankwise_guide_loss_fn = RankwiseGuideLoss()
        # self.rankwise_guide_loss_fn = PairwiseRankGuideLoss()
        self.rankwise_guide_loss_fn = ListwiseGuideLoss(cfg.T)
        self.prefix, self.postfix = template(self.llm.config.model_type, base_template=False)

        self.val_preds = []
        self.val_labels = []
        self.val_ndcg_scores = []
        self.val_recall_scores = []
        self.prepare_modules()

    @property
    def strict_loading(self):
        return False

    def prepare_modules(self):
        # Freeze the pretrained LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        # Soft tokens and classifier are trainable
        self.soft_model.soft_tokens.requires_grad = True
        for param in self.soft_model.classifier.parameters():
            param.requires_grad = True

        self.prefix_ids = self.llm_tokenizer(
            self.prefix, return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].squeeze(0)
        self.postfix_ids = self.llm_tokenizer(
            self.postfix, return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].squeeze(0)

        # Gradient checkpointing configuration
        # This is useful for fine-tuning adapter weights while keeping the model weights fixed.
        self.llm.enable_input_require_grads()
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        for module in self.llm.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.llm.train()
        return self
        # self.llm.eval()  # Ensure the base model is always in eval mode

    def on_train_start(self):
        self.llm.train()
    
    def on_train_epoch_start(self):
        self.llm.train()

    def predict_scores(self, batch):
        scores = self.soft_model(
            input_ids=batch["source_input_ids"],
            attention_mask=batch["source_attention_mask"]
        )
        return scores
    
    ### Same logic ###
    @torch.inference_mode()
    def generate_answer(self, batch, top_r_indices, batch_idx):
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
        generation_prompt_ids = batch["generation_prompt_ids"]
        answer_ids = batch["answer_ids"]
        a_len_list = batch["a_len"]
        newline_ids = self.llm_tokenizer(
            "\n\n", return_tensors="pt", add_special_tokens=False
        )["input_ids"].squeeze(0).to(doc_input_ids.device)

        # input_embeds_list = []
        input_ids_list = []
        for i in range(B):
            sample_top_r_indices = top_r_indices[i]
            sample_doc_input_ids = doc_input_ids[i]
            sample_doclen_list = doclen_list[i]
            # sample_question_ids = question_ids[i]
            sample_generation_prompt_ids = generation_prompt_ids[i]

            # SYSTEM + USER + POSTFIX
            sample_input_ids = []
            sample_input_ids.append(self.prefix_ids.to(sample_doc_input_ids.device))
            for idx in sample_top_r_indices:
                doclen = sample_doclen_list[idx]
                doc_ids = sample_doc_input_ids[idx, -doclen:]   # left padding
                sample_input_ids.append(doc_ids)
                sample_input_ids.append(newline_ids)
            sample_input_ids.append(sample_generation_prompt_ids)
            sample_input_ids.append(self.postfix_ids.to(sample_doc_input_ids.device))

            sample_input_ids = torch.cat(sample_input_ids) # (L,)
            input_ids_list.append(sample_input_ids)
            # Embedding
            # sample_input_embeds = self.llm.get_input_embeddings()(sample_input_ids)   # (L, D)
            # input_embeds_list.append(sample_input_embeds)

        # max_len = max(input_ids.shape[0] for input_ids in input_ids_list)
        padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.llm_tokenizer.pad_token_id, padding_side='left')

        # seq_lens = torch.tensor([input_ids.shape[0] for input_ids in input_ids_list], device=padded_input_ids.device)
        # mask_range = torch.arange(max_len, device=padded_input_ids.device).unsqueeze(0)
        attention_masks = [
            torch.ones(ids.shape[0], dtype=torch.long, device=ids.device) 
            for ids in input_ids_list
        ]
        padded_attention_mask = pad_sequence(attention_masks, batch_first=True, padding_value=0, padding_side='left')

        generated_ids = self.llm.generate(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )
        input_len = padded_input_ids.shape[1]
        new_token_ids = generated_ids[:, input_len:]
        
        batch_preds_text = self.llm_tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        batch_refs_text = []
        for i in range(B):
            a_len = a_len_list[i].item()
            ref_ids = answer_ids[i][:a_len]
            ref_text = self.llm_tokenizer.decode(ref_ids, skip_special_tokens=True)
            batch_refs_text.append(ref_text)

        if batch_idx == 0:
            print("=" * 60)
            print(f"Input length: {input_len}, Output length: {generated_ids.shape[1]}")
            print(f"New tokens: {new_token_ids.shape}")
            
            # 사람이 읽을 수 있는 형태로 input 확인
            decoded_input = self.llm_tokenizer.decode(
                input_ids_list[0], skip_special_tokens=False
            )
            print(f"Input (sample 0):\n{decoded_input}")
            
            # 생성된 답 확인
            print(f"Predicted: {batch_preds_text[0]}")
            print(f"Reference: {batch_refs_text[0]}")
            print("=" * 60)
        
        return batch_preds_text, batch_refs_text

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
        batch_recall = []
        for i in range(B):
            sample_ndcg = compute_ndcg(scores_hat_np[i], scores_oracle_np[i])
            sample_recall = compute_recall(scores_hat_np[i], scores_oracle_np[i])
            batch_ndcg.append(sample_ndcg)
            batch_recall.append(sample_recall)

        topr_indices = topk_student_indices[:, :self.top_r]
        return topr_indices, batch_ndcg, batch_recall
    ### Same logic ###


    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            logger.info(f"Train Step {batch_idx}:")
            logger.info(f"self.training: {self.training}")
            logger.info(f"self.soft_model.training: {self.soft_model.training}")
            logger.info(f"self.soft_model.llm.training: {self.soft_model.llm.training}")
            logger.info(f"self.soft_model.llm.model.training: {self.soft_model.llm.model.training}")
            logger.info(f"gradient checkpointing: {self.llm.is_gradient_checkpointing}")

        scores_hat = self.predict_scores(batch)  # (B*K,)

        raw_scores_oracle = batch["scores_oracle"].reshape(-1, 1)   # (B*K, 1)
        scores_oracle = raw_scores_oracle.to(scores_hat.device)

        # Guide loss (Pointwise + Rankwise)
        scores_hat = scores_hat.view(-1).to(scores_oracle.dtype)
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

        # Debugging logs every 10 steps
        accumulate = self.trainer.accumulate_grad_batches
        if batch_idx % (10 * accumulate) == 0:
            pred_per_query = scores_hat.reshape(B, K).detach().cpu().float().numpy()
            oracle_per_query = scores_oracle.reshape(B, K).detach().cpu().float().numpy()
            rhos = []
            for i in range(B):
                rho, _ = spearmanr(pred_per_query[i], oracle_per_query[i])
                if not np.isnan(rho):
                    rhos.append(rho)
            avg_rho = np.mean(rhos) if rhos else 0.0

            logger.info(f"Step {self.global_step}:")
            logger.info(f"  pred_scores std: {scores_hat.reshape(B, K).std(dim=-1).mean().item():.4f}")
            logger.info(f"  pred_scores range: [{scores_hat.min().item():.4f}, {scores_hat.max().item():.4f}]")
            logger.info(f"  oracle_scores std: {scores_oracle.reshape(B, K).std(dim=-1).mean().item():.4f}")
            logger.info(f"  oracle_scores range: [{scores_oracle.min().item():.4f}, {scores_oracle.max().item():.4f}]")
            logger.info(f"  Spearman's rho (pred vs oracle): {avg_rho:.4f}")
            logger.info(f"  loss: {loss.item():.4f}")

            self.log("train/spearman_rho", avg_rho, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            logger.info(f"Validation Step {batch_idx}:")
            logger.info(f"self.training: {self.training}")
            logger.info(f"self.llm.training: {self.llm.training}")
            logger.info(f"self.llm.model.training: {self.llm.model.training}")
            logger.info(f"gradient checkpointing: {self.llm.is_gradient_checkpointing}")
        
        scores_hat = self.predict_scores(batch)  # (B*K,)

        raw_scores_oracle = batch["scores_oracle"].reshape(-1, 1)   # (B*K, 1)
        scores_oracle = raw_scores_oracle.to(scores_hat.device)

        # Guide loss (Pointwise + Rankwise)
        scores_hat = scores_hat.view(-1).to(scores_oracle.dtype)
        scores_oracle = scores_oracle.view(-1)
        raw_pointwise_guide_loss = self.pointwise_guide_loss_fn(scores_hat, scores_oracle)
        B, K = batch["doclen_list"].shape
        raw_rankwise_guide_loss = self.rankwise_guide_loss_fn(scores_hat.reshape(B, K), scores_oracle.reshape(B, K))
        
        pointwise_guide_loss = (1 - self.alpha) * raw_pointwise_guide_loss
        rankwise_guide_loss = self.alpha * raw_rankwise_guide_loss
        loss = pointwise_guide_loss + rankwise_guide_loss
        
        # TODO: Re-ranking, select Top-R
        top_r_indices, batch_ndcg, batch_recall = self.reranking_documents(batch, scores_hat, raw_scores_oracle)
        self.val_ndcg_scores.extend(batch_ndcg)
        self.val_recall_scores.extend(batch_recall)

        # Generation quality evaluation
        batch_preds, batch_labels = self.generate_answer(batch, top_r_indices, batch_idx)
        self.val_preds.extend(batch_preds)
        self.val_labels.extend(batch_labels)

        # self.log("valid/oracle_score", scores_oracle, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/pointwise_guide_loss", raw_pointwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid/rankwise_guide_loss", raw_rankwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

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
        np_recall = np.array(self.val_recall_scores)
        avg_recall = np.mean(np_recall, axis=0)

        self.log("valid/EM", avg_em, sync_dist=True)
        self.log("valid/F1", avg_f1, sync_dist=True)
        self.log("valid/NDCG@1", avg_ndcg[0], sync_dist=True)
        self.log("valid/NDCG@3", avg_ndcg[1], sync_dist=True)
        self.log("valid/NDCG@5", avg_ndcg[2], sync_dist=True)
        self.log("valid/NDCG@10", avg_ndcg[3], sync_dist=True)
        self.log("valid/Recall@1", avg_recall[0], sync_dist=True)
        self.log("valid/Recall@3", avg_recall[1], sync_dist=True)
        self.log("valid/Recall@5", avg_recall[2], sync_dist=True)
        self.log("valid/Recall@10", avg_recall[3], sync_dist=True)

        self.val_preds.clear()
        self.val_labels.clear()
        self.val_ndcg_scores.clear()
        self.val_recall_scores.clear()
        
    def on_save_checkpoint(self, checkpoint):
        # save only CAFormer classifier weights
        state_dict = checkpoint["state_dict"]
        soft_state = {
            k: v for k, v in state_dict.items() if "soft_model" in k and "llm" not in k
        }
        checkpoint["state_dict"] = soft_state

    def configure_optimizers(self):
        # Soft tokens: Lower LR
        # Classifier: Larger LR (random init)
        soft_token_params = [self.soft_model.soft_tokens]
        classifier_params = [p for p in self.soft_model.classifier.parameters() if p.requires_grad]

        optimizer_grouped_parameters = [
            {"params": soft_token_params, "lr": self.learning_rate},
            {"params": classifier_params, "lr": 1e-3}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, fused=False)

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
    
    # Debugging: Log gradient norms of CAFormer and classifier every 10 steps
    def on_after_backward(self):
        if self.global_step % 10 == 0:
            soft_grad_norm = self.soft_model.soft_tokens.grad.norm(2).item() \
                if self.soft_model.soft_tokens.grad is not None else 0.0
            cls_grad_norm = self._compute_grad_norm(self.soft_model.classifier)
            self.log("train/soft_tokens_grad_norm", soft_grad_norm, on_step=True)
            self.log("train/classifier_grad_norm", cls_grad_norm, on_step=True)

    def _compute_grad_norm(self, module):
        total_norm_sq = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        return total_norm_sq ** 0.5