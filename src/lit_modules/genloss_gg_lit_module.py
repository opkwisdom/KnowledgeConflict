import torch.nn as nn
import torch
import logging
import re
from torch.nn.utils.rnn import pad_sequence
from typing import Union
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM

from utils import compute_metrics, parse_reference_answer
from models import load_model, CAFormerGGClassifier, RankwiseGuideLoss, PairwiseRankGuideLoss

logger = logging.getLogger(__name__)


class GenLossGGLightningModule(LightningModule):
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

        self.val_logits = []
        self.val_labels = []
        self.prepare_modules()

    def prepare_modules(self):
        # Freeze the pretrained LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        
        for param in self.caformer_clf.parameters():
            param.requires_grad = True
            
        # Gradient checkpointing configuration
        # This is useful for fine-tuning adapter weights while keeping the model weights fixed.
        self.llm.enable_input_require_grads()
        self.llm.gradient_checkpointing_enable()
        if hasattr(self.llm.config, "attention_dropout"):
            self.llm.config.attention_dropout = 0.0
        if hasattr(self.llm.config, "dropout"):
            self.llm.config.dropout = 0.0

    def train(self, mode: bool = True):
        super().train(mode)
        # self.llm.eval()  # Ensure the base model is always in eval mode

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
    
    def compute_interleaving_loss(self, batch, query_hidden_states, top_r_indices, return_logits=False):
        """
        Compute the generation loss conditioned on the interleaving document inputs.
        This is computed using only re-ranked docs(top_r_indices).
        interleaving document inputs: D_1 + Q_{CA}^{(1)} + ... + D_K + Q_{CA}^{(K)} + Q + A (Teacher-forcing)
        Args:
            batch: The input batch containing document, question, and answer IDs.
            query_hidden_states: Tensor of shape (B*k, S, D_llm)
            top_r_indices: Tensor of shape (B, R)
        Returns:
            gen_loss: scalar tensor
            (Optional) batch_logits: length of B List of Tensor of shape (A,), only Top-1
            (Optional) batch_labels: length of B List of Tensor of shape (A,)
        """
        # Meta-infos
        doclen_list = batch["doclen_list"]  # (B, K)
        a_len_list = batch["a_len"]         # (B,)
        B, K = doclen_list.shape
        S = query_hidden_states.shape[1]
        
        # These things will be re-combined
        query_hidden_states = query_hidden_states.reshape(B, K, S, -1)
        doc_input_ids = batch["doc_input_ids"].reshape(B, K, -1)
        question_ids = batch["question_ids"]
        answer_ids = batch["answer_ids"]

        input_embeds_list = []
        attention_mask_list = []
        labels_list = []
        
        for i, sample_top_r_indices in enumerate(top_r_indices):
            sample_query_hidden_states = query_hidden_states[i]
            sample_doc_input_ids = doc_input_ids[i]
            sample_doclen_list = doclen_list[i]
            sample_question_ids = question_ids[i]
            sample_answer_ids = answer_ids[i]
            sample_a_len = a_len_list[i]

            sample_selected_qh = [
                sample_query_hidden_states[idx]
                for idx in sample_top_r_indices
            ]
            sample_input_ids = [
                sample_doc_input_ids[idx]
                for idx in sample_top_r_indices
            ]
            sample_doclens = [
                sample_doclen_list[idx]
                for idx in sample_top_r_indices
            ]
            
            sample_input_ids = torch.cat(sample_input_ids) # (L,)
            sample_input_ids = torch.cat([sample_input_ids, sample_question_ids, sample_answer_ids])
            sample_raw_embeds = self.llm.get_input_embeddings()(sample_input_ids)   # (L, D)

            # Combine
            sample_input_embeds = []
            start_pos = 0
            for doclen, qh_emb in zip(sample_doclens, sample_selected_qh):
                doc_emb = sample_raw_embeds[start_pos:start_pos + doclen]
                sample_input_embeds.append(doc_emb)
                sample_input_embeds.append(qh_emb)
                start_pos += doclen
            sample_input_embeds.append(sample_raw_embeds[start_pos:])   # Append QA embeddings at the end
            sample_input_embeds = torch.cat(sample_input_embeds, dim=0)
            
            # label construction
            total_len = sample_input_embeds.shape[0]
            sample_labels = torch.full((total_len,), -100, dtype=torch.long, device=sample_input_embeds.device)
            sample_labels[-sample_a_len:] = sample_answer_ids
            
            input_embeds_list.append(sample_input_embeds)
            labels_list.append(sample_labels)

        # Pad manually
        max_len = max(input_embeds.shape[0] for input_embeds in input_embeds_list)
        padded_input_embeds = pad_sequence(input_embeds_list, batch_first=True, padding_value=0.0)
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        seq_lens = torch.tensor([input_embeds.shape[0] for input_embeds in input_embeds_list], device=padded_input_embeds.device)
        mask_range = torch.arange(max_len, device=padded_input_embeds.device).unsqueeze(0)
        padded_attention_mask = (mask_range < seq_lens.unsqueeze(1)).long()
        
        # Generate logits and compute loss
        # GPU MEMORY BOTTLENECK!! (~13GB for 10 docs)
        outputs = self.llm.model(
            inputs_embeds=padded_input_embeds,
            attention_mask=padded_attention_mask,
            use_cache=False
        )
        hidden_states = outputs.last_hidden_state   # (B, L_total, D_llm)
        
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = padded_labels[..., 1:].contiguous()
        active_loss_mask = shift_labels != -100

        valid_hidden = shift_hidden[active_loss_mask]
        valid_labels = shift_labels[active_loss_mask]
        
        valid_logits = self.llm.lm_head(valid_hidden)
        gen_loss = self.gen_loss_fn(valid_logits, valid_labels)

        batch_logits, batch_labels = [], []
        if return_logits:
            all_preds = valid_logits.argmax(dim=-1).detach().cpu()
            all_labels = valid_labels.detach().cpu()
            # all_labels = [gold_answer for gold_answer in batch["gold_answer"]]
            curr = 0
            for i in range(B):
                length = a_len_list[i].item()
                batch_logits.append(all_preds[curr:curr+length])
                batch_labels.append(all_labels[curr:curr+length])
                curr += length
        
        outputs = (gen_loss,)
        if return_logits:
            outputs += (batch_logits, batch_labels)
            
        return outputs
    
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
        lmbda = self.get_reranking_scheduled_weight()
        
        scores = (1 - lmbda) * scores_oracle_reshaped + lmbda * scores_hat_reshaped
        topk_indices = scores.sort(dim=-1, descending=True)[1]    # (B, K)
        topr_indices = topk_indices[:, :self.top_r]
        return topr_indices

    def training_step(self, batch, batch_idx):
        llm_repr = self.forward(batch)
        scores_hat, query_hidden_states = self.caformer_clf(llm_repr, batch["source_attention_mask"],
                                                            batch["roberta_question_ids"], batch["roberta_question_mask"])
        del llm_repr

        scores_oracle = batch["scores_oracle"].reshape(-1, 1)   # (B*K, 1)
        scores_oracle = scores_oracle.to(scores_hat.device)
        scores_oracle = self.transform_scores_oracle(scores_oracle)

        # Guide loss (Pointwise + Rankwise)
        scores_hat = scores_hat.view(-1)
        scores_oracle = scores_oracle.view(-1)
        raw_pointwise_guide_loss = self.pointwise_guide_loss_fn(scores_hat, scores_oracle)
        B, K = batch["doclen_list"].shape
        raw_rankwise_guide_loss = self.rankwise_guide_loss_fn(scores_hat.reshape(B, K), scores_oracle.reshape(B, K))
        
        pointwise_guide_loss = (1 - self.alpha) * raw_pointwise_guide_loss
        rankwise_guide_loss = self.alpha * raw_rankwise_guide_loss
        guide_loss = pointwise_guide_loss + rankwise_guide_loss

        # TODO: Re-ranking, select Top-R
        # Re-rank documents
        top_r_indices = self.reranking_documents(batch, scores_hat, scores_oracle)

        # Generation loss, using re-ranked docs
        gen_loss = self.compute_interleaving_loss(batch, query_hidden_states, top_r_indices)[0]
        loss = self.gamma * guide_loss + gen_loss
        
        # self.log("train/oracle_score", scores_oracle, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/pointwise_guide_loss", self.gamma * raw_pointwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/rankwise_guide_loss", self.gamma * raw_rankwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/guide_loss", self.gamma * guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/gen_loss", gen_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        # Debugging
        if batch_idx % 100 == 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"[Val Epoch / Global Step: {self.global_step}] 점수 분포 분석")
            logger.info(f"{'='*50}")
            
            # 1. Oracle Score (정답) 통계
            logger.info(f"[Target: scores_oracle]")
            logger.info(f"   Mean: {scores_oracle.mean().item():.6f} | Std : {scores_oracle.std().item():.6f}")
            logger.info(f"   Min : {scores_oracle.min().item():.6f} | Max : {scores_oracle.max().item():.6f}")
            
            # 2. Hat Score (예측) 통계
            logger.info(f"[Prediction: scores_hat]")
            logger.info(f"   Mean: {scores_hat.mean().item():.6f} | Std : {scores_hat.std().item():.6f}")
            logger.info(f"   Min : {scores_hat.min().item():.6f} | Max : {scores_hat.max().item():.6f}")
            logger.info(f"{'-'*50}")
            
            # 3. 샘플 값 직접 눈으로 비교 (하위 5개 / 상위 5개)
            # 타겟을 기준으로 정렬하여, 타겟이 낮을/높을 때 모델의 예측값이 어떻게 따라가는지 확인
            sorted_oracle, sorted_indices = torch.sort(scores_oracle)
            matched_hat = scores_hat[sorted_indices]
            
            logger.info(f"[하위 5개 샘플]")
            logger.info(f"   Oracle: {sorted_oracle[:5].detach().cpu().float().numpy()}")
            logger.info(f"   Hat   : {matched_hat[:5].detach().cpu().float().numpy()}")
            
            logger.info(f"[상위 5개 샘플]")
            logger.info(f"   Oracle: {sorted_oracle[-5:].detach().cpu().float().numpy()}")
            logger.info(f"   Hat   : {matched_hat[-5:].detach().cpu().float().numpy()}")
            logger.info(f"{'='*50}\n")

        return loss

    def validation_step(self, batch, batch_idx):
        llm_repr = self.forward(batch)
        scores_hat, query_hidden_states = self.caformer_clf(llm_repr, batch["source_attention_mask"],
                                                            batch["roberta_question_ids"], batch["roberta_question_mask"])
        del llm_repr

        # Compute the gradient of the oracle loss and target score
        scores_oracle = batch["scores_oracle"].reshape(-1, 1)   # (B*K, 1)
        scores_oracle = scores_oracle.to(scores_hat.device)
        scores_oracle = self.transform_scores_oracle(scores_oracle)

        # Guide loss (Pointwise + Rankwise)
        scores_hat = scores_hat.view(-1)
        scores_oracle = scores_oracle.view(-1)
        raw_pointwise_guide_loss = self.pointwise_guide_loss_fn(scores_hat, scores_oracle)
        B, K = batch["doclen_list"].shape
        raw_rankwise_guide_loss = self.rankwise_guide_loss_fn(scores_hat.reshape(B, K), scores_oracle.reshape(B, K))
        
        pointwise_guide_loss = (1 - self.alpha) * raw_pointwise_guide_loss
        rankwise_guide_loss = self.alpha * raw_rankwise_guide_loss
        guide_loss = pointwise_guide_loss + rankwise_guide_loss
        
        # TODO: Re-ranking, select Top-R
        # Re-rank documents
        top_r_indices = self.reranking_documents(batch, scores_hat, scores_oracle)

        # Generation loss, using re-ranked docs
        gen_loss, batch_logits, batch_labels = self.compute_interleaving_loss(batch, query_hidden_states, top_r_indices, return_logits=True)
        loss = self.gamma * guide_loss + gen_loss
        
        self.val_logits.extend(batch_logits)
        self.val_labels.extend(batch_labels)

        # self.log("valid/oracle_score", scores_oracle, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/pointwise_guide_loss", self.gamma * raw_pointwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid/rankwise_guide_loss", self.gamma * raw_rankwise_guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid/guide_loss", self.gamma * guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid/gen_loss", gen_loss, on_step=False, on_epoch=True, sync_dist=True)
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
        
        # Compute EM, F1 based on val_logits and val_labels
        for logits, labels in zip(self.val_logits, self.val_labels):
            pred = self.llm_tokenizer.decode(logits, skip_special_tokens=True)
            answer = self.llm_tokenizer.decode(labels, skip_special_tokens=True)
            # gold_answers = parse_reference_answer(answer)
            
            pred_short = self.parse_final_answer(pred)
            gold_short = self.parse_final_answer(answer)
            metrics = compute_metrics(pred_short, gold_short)

            total_em += float(metrics.soft_em)
            total_f1 += float(metrics.f1)
            total_count += 1

        avg_em = total_em / total_count if total_count > 0 else 0.0
        avg_f1 = total_f1 / total_count if total_count > 0 else 0.0
        
        self.log("valid/EM", avg_em, sync_dist=True)
        self.log("valid/F1", avg_f1, sync_dist=True)

        self.val_logits.clear()
        self.val_labels.clear()
        
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
            p for p in self.caformer_clf.ca_former.parameters() if p.requires_grad
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
    
    def get_reranking_scheduled_weight(self):
        """
        Linearly teacher forcing
        """
        if not self.training:
            return 1.0
        
        current_step = self.global_step
        total_steps = self.trainer.estimated_stepping_batches
        max_epochs = self.trainer.max_epochs
        
        if total_steps == float('inf') or max_epochs is None:
            return 1.0
        
        steps_per_epoch = total_steps / max_epochs
        
        phase1_end_step = steps_per_epoch * 0.2
        phase2_end_step = steps_per_epoch * 0.8
        
        # ~20% of epoch 1
        if current_step < phase1_end_step:
            return 0.0
        # 20% ~ 80% of epoch 1
        elif current_step < phase2_end_step:
            progress = (current_step - phase1_end_step) / (phase2_end_step - phase1_end_step)
            return float(progress)
        else:
            return 1.0