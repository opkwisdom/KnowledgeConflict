import torch.nn as nn
import torch
import logging
import wandb
import torch.nn.functional as F
from typing import Union
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup, RobertaModel

from models import MultiHiddenCAFormer, MultiQueryContrastiveLoss, load_model, CosSimRegLoss

logger = logging.getLogger(__name__)

class CTLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig, llm: nn.Module, caformer: MultiHiddenCAFormer, roberta: RobertaModel):
        super().__init__()
        self.save_hyperparameters(ignore=["llm", "caformer", "roberta"])

        self.llm = llm
        self.caformer = caformer
        self.roberta = roberta  # reference roberta model
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        # self.mask_schedule = cfg.mask_schedule

        self.ctr_loss_fn = MultiQueryContrastiveLoss(T=getattr(cfg, "temperature", None) or 1.0)
        self.cos_reg_loss_fn = CosSimRegLoss()
        self.prepare_modules()

    def prepare_modules(self):
        # Freeze the pretrained modules (Roberta, LLM)
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        for param in self.caformer.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()
        self.roberta.eval()

    # def _compute_mask_ratio(self):
    #     current_step_ratio = self.global_step / self.trainer.estimated_stepping_batches
    #     if current_step_ratio < self.mask_schedule.warmup_ratio:
    #         return 0.0      # No masking at the beginning
    #     else:
    #         # Linear interpolation between start and end mask ratio
    #         progress = (current_step_ratio - self.mask_schedule.start_ratio) / \
    #                    (1.0 - self.mask_schedule.start_ratio)
    #         return progress * self.mask_schedule.target_mask_ratio

    # def get_masked_input(self, batch):
    #     pass

    def forward(self, batch):
        """
        Forward pass through the model Roberta and DISCA.
        """
        roberta_inputs, llm_inputs = batch["roberta"], batch["llm"]
        # Get Roberta hidden states
        with torch.no_grad():
            roberta_outputs = self.roberta(**roberta_inputs, output_hidden_states=True)
            roberta_repr = roberta_outputs.last_hidden_state.mean(dim=1)   # mean pooling
        # Get LLM hidden states
        with torch.no_grad():
            llm_outputs = self.llm(**llm_inputs, output_hidden_states=True)
            llm_repr = torch.stack(llm_outputs.hidden_states).permute(1, 0, 2, 3)   # (B, L, S, D_llm)
        
        return roberta_repr, llm_repr

    def training_step(self, batch, batch_idx):
        roberta_repr, llm_repr = self.forward(batch)
        caformer_input = {
            "llm_hidden_states": llm_repr,
            "attention_mask": batch["llm"]["attention_mask"]
        }
        caformer_repr = self.caformer(**caformer_input)[0]
        B, K, D = caformer_repr.shape

        cos_reg_loss = torch.tensor(0.0, device=caformer_repr.device)
        if self.trainer.world_size > 1:
            gathered_caformer_repr = self.all_gather(caformer_repr, sync_grads=True).reshape(-1, K, D)
            gathered_roberta_repr = self.all_gather(roberta_repr, sync_grads=True).reshape(-1, D)
            max_loss, mean_loss, _ = self.ctr_loss_fn(gathered_caformer_repr, gathered_roberta_repr)
            # cosine similarity regularization
            if getattr(self.cfg, "lambda_c", 0) > 0:
                cos_reg_loss = self.cos_reg_loss_fn(gathered_caformer_repr)
        else:
            max_loss, mean_loss, _ = self.ctr_loss_fn(caformer_repr, roberta_repr)
            # cosine similarity regularization
            if getattr(self.cfg, "lambda_c", 0) > 0:
                cos_reg_loss = self.cos_reg_loss_fn(caformer_repr)
        loss = max_loss + self.cfg.ctr_loss.lambda_m * mean_loss + self.cfg.lambda_c * cos_reg_loss


        self.log("train/max_ctr_loss", max_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mean_ctr_loss", mean_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/ctr_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        roberta_repr, llm_repr = self.forward(batch)
        caformer_input = {
            "llm_hidden_states": llm_repr,
            "attention_mask": batch["llm"]["attention_mask"]
        }
        caformer_repr = self.caformer(**caformer_input)[0]
        B, K, D = caformer_repr.shape

        if self.trainer.world_size > 1:
            gathered_caformer_repr = self.all_gather(caformer_repr, sync_grads=True).reshape(-1, K, D)
            gathered_roberta_repr = self.all_gather(roberta_repr, sync_grads=True).reshape(-1, D)
            max_loss, mean_loss, val_acc = self.ctr_loss_fn(gathered_caformer_repr, gathered_roberta_repr)
        else:
            max_loss, mean_loss, val_acc = self.ctr_loss_fn(caformer_repr, roberta_repr)
        loss = max_loss + self.cfg.ctr_loss.lambda_m * mean_loss

        self.log("valid/max_ctr_loss", max_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid/mean_ctr_loss", mean_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid/ctr_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid/acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def on_save_checkpoint(self, checkpoint):
        # save only CAFormer weights
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