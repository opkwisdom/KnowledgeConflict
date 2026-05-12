import torch.nn as nn
import torch
import logging
import wandb
import torch.nn.functional as F
from typing import Union
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from models import MultiHiddenCAFormer, load_model

logger = logging.getLogger(__name__)

class RCLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig, llm: nn.Module, llm_tokenizer: AutoTokenizer, caformer: MultiHiddenCAFormer):
        super().__init__()
        self.save_hyperparameters(ignore=["llm", "caformer"])

        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.caformer = caformer
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate

        self.prepare_modules()

    def prepare_modules(self):
        # Freeze the pretrained modules (LLM)
        for param in self.llm.parameters():
            param.requires_grad = False
        
        for param in self.caformer.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()

    def forward(self, batch):
        # Get LLM representations
        with torch.no_grad():
            llm_outputs = self.llm(**batch, output_hidden_states=True)
            llm_repr = torch.stack(llm_outputs.hidden_states).permute(1, 0, 2, 3)   # (B, L, S, D_llm)
        
        # Get CAFormer output
        caformer_input = {
            "llm_hidden_states": llm_repr,
            "attention_mask": batch["attention_mask"]
        }
        caformer_repr = self.caformer(**caformer_input)[1]  # (B, K, D_llm)

        return caformer_repr

    def training_step(self, batch, batch_idx):
        caformer_repr = self.forward(batch)

        # Right-hand side LLM representations
        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(batch["input_ids"])
        inputs_embeds = torch.cat([caformer_repr, text_embeds], dim=1)  # (B, K+S, D_llm)
        B, K, _ = caformer_repr.size()
        query_mask = torch.ones((B, K), dtype=batch["attention_mask"].dtype, device=self.device)
        attention_mask = torch.cat([query_mask, batch["attention_mask"]], dim=1)

        # Prepare labels for NLLloss
        text_labels = batch["input_ids"].clone()
        text_labels[text_labels == self.llm_tokenizer.pad_token_id] = -100  # Ignore padding tokens
        ignore_labels = torch.full((B, K), -100, dtype=text_labels.dtype, device=text_labels.device)
        labels = torch.cat([ignore_labels, text_labels], dim=1) # (B, K+S)

        # Get LLM output logits
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = llm_outputs.loss
        
        # Diagnostic logging (K query diversity)
        with torch.no_grad():
            normed = F.normalize(caformer_repr, p=2, dim=-1)
            sim_matrix = torch.bmm(normed, normed.transpose(1, 2))  # (B, K, K)
            eye_mask = torch.eye(K, dtype=torch.bool, device=caformer_repr.device).unsqueeze(0)
            off_diag = sim_matrix.masked_fill(eye_mask, 0.0)
            cos_sim_mean = off_diag.sum() / (B * K * (K - 1))
            cos_sim_max = off_diag.max()
            
        self.log("train/nll_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/query_cos_sim_mean", cos_sim_mean, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/query_cos_sim_max", cos_sim_max, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        caformer_repr = self.forward(batch)

        # Right-hand side LLM representations
        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(batch["input_ids"])
        inputs_embeds = torch.cat([caformer_repr, text_embeds], dim=1)  # (B, K+S, D_llm)
        B, K, _ = caformer_repr.size()
        query_mask = torch.ones((B, K), dtype=batch["attention_mask"].dtype, device=self.device)
        attention_mask = torch.cat([query_mask, batch["attention_mask"]], dim=1)

        # Prepare labels for NLLloss
        text_labels = batch["input_ids"].clone()
        text_labels[text_labels == self.llm_tokenizer.pad_token_id] = -100  # Ignore padding tokens
        ignore_labels = torch.full((B, K), -100, dtype=text_labels.dtype, device=text_labels.device)
        labels = torch.cat([ignore_labels, text_labels], dim=1) # (B, K+S)

        # Get LLM output logits
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = llm_outputs.loss
        
        # Diagnostic logging (K query diversity)
        with torch.no_grad():
            normed = F.normalize(caformer_repr, p=2, dim=-1)
            sim_matrix = torch.bmm(normed, normed.transpose(1, 2))  # (B, K, K)
            eye_mask = torch.eye(K, dtype=torch.bool, device=caformer_repr.device).unsqueeze(0)
            off_diag = sim_matrix.masked_fill(eye_mask, 0.0)
            cos_sim_mean = off_diag.sum() / (B * K * (K - 1))
            cos_sim_max = off_diag.max()

        # Token-level accuracy
        logits = llm_outputs.logits     # (B, K+S, Vocab)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        preds = torch.argmax(shift_logits, dim=-1)
        valid_mask = shift_labels != -100
        correct_preds = (preds == shift_labels) & valid_mask
        val_acc = correct_preds.sum().float() / valid_mask.sum().float()

        self.log("valid/nll_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid/acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid/query_cos_sim_mean", cos_sim_mean, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("valid/query_cos_sim_max", cos_sim_max, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss


    def on_save_checkpoint(self, checkpoint):
        # Save only CAFormer weights
        state_dict = checkpoint["state_dict"]
        caformer_state_dict = {
            k: v for k, v in state_dict.items() if "caformer" in k
        }
        checkpoint["state_dict"] = caformer_state_dict

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.caformer.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
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
                "frequency": 1
            }
        }