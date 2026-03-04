import torch.nn as nn
import torch
import logging
import wandb
from typing import Union
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from torchmetrics import ConfusionMatrix, F1Score, Accuracy
from transformers import get_linear_schedule_with_warmup

from models import DISCA, CAFormerClassifier, ContrastiveLoss

logger = logging.getLogger(__name__)


class CAFormerLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig, disca: DISCA, caformer_clf: CAFormerClassifier):
        super().__init__()
        self.save_hyperparameters(ignore=["disca", "caformer_clf"])

        self.disca = disca
        self.caformer_clf = caformer_clf
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate

        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.ctr_loss_fn = ContrastiveLoss(T=getattr(cfg, "temperature", None) or 1.0)
        self.ctr_loss_weight = getattr(cfg, "ctr_loss_weight", 1.0)
        self.prepare_modules()

        # Metrics
        self.acc = Accuracy(task="multiclass", num_classes=3)
        self.f1 = F1Score(task="multiclass", num_classes=3, average=None)
        self.conf_mat = ConfusionMatrix(task="multiclass", num_classes=3)
        self.val_results = []

    def prepare_modules(self):
        for param in self.disca.model.parameters():
            param.requires_grad = False
        for param in self.caformer_clf.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.disca.model.eval()  # Ensure the base model is always in eval mode

    def forward(self, batch):
        queries = batch["queries"]
        ctxs_list = batch["ctxs_list"]

        with torch.no_grad():
            caformer_inputs, caformer_masks = self.disca(queries, ctxs_list)
        return self.caformer_clf(caformer_inputs, caformer_masks)

    def training_step(self, batch, batch_idx):
        labels_tensor = batch["labels_tensor"]
        logits, pooled_output = self(batch)

        ce_loss = self.ce_loss_fn(logits, labels_tensor)
        if self.trainer.world_size > 1:
            gathered_pooled = self.all_gather(pooled_output, sync_grads=True)
            gathered_labels = self.all_gather(labels_tensor, sync_grads=True)
            
            gathered_pooled = gathered_pooled.view(-1, gathered_pooled.shape[-1])
            gathered_labels = gathered_labels.view(-1)
            ctr_loss = self.ctr_loss_fn(gathered_pooled, gathered_labels)
        else:
            ctr_loss = self.ctr_loss_fn(pooled_output, labels_tensor)
        loss = ce_loss + self.ctr_loss_weight * ctr_loss

        acc = self.acc(logits, labels_tensor)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels_tensor = batch["labels_tensor"]
        logits, pooled_output = self(batch)

        ce_loss = self.ce_loss_fn(logits, labels_tensor)
        ctr_loss = self.ctr_loss_fn(pooled_output, labels_tensor)
        loss = ce_loss + self.ctr_loss_weight * ctr_loss
        
        preds = torch.argmax(logits, dim=1)
        self.val_results.append({
            "preds": preds,
            "targets": labels_tensor,
        })
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        preds = torch.cat([res["preds"] for res in self.val_results])
        targets = torch.cat([res["targets"] for res in self.val_results])

        # all gather
        all_preds = self.all_gather(preds).flatten()
        all_targets = self.all_gather(targets).flatten()

        f1_scores = self.f1(all_preds, all_targets)
        acc_scores = self.acc(all_preds, all_targets)
        self.log("val_f1_sup", f1_scores[0], prog_bar=True)
        self.log("val_f1_ctd", f1_scores[1], prog_bar=True)
        self.log("val_f1_irr", f1_scores[2], prog_bar=True)
        self.log("val_acc", acc_scores, prog_bar=True)

        if self.trainer.is_global_zero:
            preds_np = all_preds.cpu().numpy()
            targets_np = all_targets.cpu().numpy()

            wandb_logger = None
            if isinstance(self.logger, list):
                for logger in self.logger:
                    if "wandb" in str(type(logger)).lower():
                        wandb_logger = logger
                        break
            else:
                wandb_logger = self.logger
            
            conf_mat_plot = wandb.plot.confusion_matrix(
                probs=None,
                y_true=targets_np,
                preds=preds_np,
                class_names=["Supportive", "Contradictory", "Irrelevant"],
                title=f"Confusion Matrix (Epoch {self.current_epoch})"
            )
            wandb_logger.experiment.log({"val_cm": conf_mat_plot, "epoch": self.current_epoch})

        self.val_results.clear()
        self.acc.reset()
        self.f1.reset()
        self.conf_mat.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.caformer_clf.parameters(), lr=self.learning_rate)
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