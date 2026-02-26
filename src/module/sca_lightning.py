import torch.nn as nn
import torch
import logging
import wandb
from typing import Union
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from torchmetrics import ConfusionMatrix, F1Score, Accuracy
from transformers import get_linear_schedule_with_warmup

from models import DISCA, SimpleDISCA, KVFormer

logger = logging.getLogger(__name__)


class SCALightningModule(LightningModule):
    def __init__(self, cfg: DictConfig, disca: Union[DISCA, SimpleDISCA], kv_former: KVFormer):
        super().__init__()
        self.save_hyperparameters(ignore=["disca", "kv_former"])

        self.disca = disca
        self.kv_former = kv_former
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.prepare_modules()

    def prepare_modules(self):
        self.disca.eval()
        for param in self.disca.parameters():
            param.requires_grad = False
        self.kv_former.train()

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
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