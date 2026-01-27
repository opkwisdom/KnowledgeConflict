from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import logging
import wandb
from omegaconf import DictConfig
from torchmetrics import ConfusionMatrix, F1Score, Accuracy
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ConflictDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.input_dim = cfg.n_heads * cfg.n_features
        self.input_norm = nn.LayerNorm(self.input_dim)
        # Projection layer
        self.proj = nn.Linear(self.input_dim, cfg.inter_channels)
        self.relu = nn.ReLU()

        # Feature extractor
        self.conv1 = nn.Conv1d(
            in_channels=cfg.inter_channels, out_channels=cfg.inter_channels,
            kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(cfg.inter_channels)
        self.conv2 = nn.Conv1d(
            in_channels=cfg.inter_channels, out_channels=cfg.out_channels,
            kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(cfg.out_channels)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(cfg.out_channels * cfg.n_layer, 3)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Layer, Heads, Features)
        Returns:
            logits: Tensor of shape (Batch, 3)
        """
        x = x.to(dtype=torch.float32)   # BF16 -> FP32
        B, L, H, F = x.shape
        x = x.reshape(B, L, H * F)
        # Projection
        x = self.input_norm(x)
        x = self.relu(self.proj(x))
        # Feature extraction
        x = x.permute(0, 2, 1)  # (B, inter_dim, L)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)
    

class ConflictDetectorModule(LightningModule):
    def __init__(self, cfg: DictConfig, model: ConflictDetector):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.model.train()
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.acc = Accuracy(task="multiclass", num_classes=3)
        self.f1 = F1Score(task="multiclass", num_classes=3, average=None)
        self.conf_mat = ConfusionMatrix(task="multiclass", num_classes=3)
        self.val_results = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        acc = self.acc(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)

        preds = torch.argmax(outputs, dim=1)
        self.val_results.append({
            "preds": preds.cpu(),
            "targets": targets.cpu(),
        })
        return loss
    
    def on_validation_epoch_end(self):
        all_preds = torch.cat([res["preds"] for res in self.val_results], dim=0)
        all_targets = torch.cat([res["targets"] for res in self.val_results], dim=0)

        f1_scores = self.f1(all_preds.to(self.device), all_targets.to(self.device))
        acc_scores = self.acc(all_preds.to(self.device), all_targets.to(self.device))
        self.log("val_f1_non_conflict", f1_scores[0], prog_bar=True)
        self.log("val_f1_negative", f1_scores[1], prog_bar=True)
        self.log("val_f1_positive", f1_scores[2], prog_bar=True)
        self.log("val_acc", acc_scores, prog_bar=True)

        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()

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
            class_names=["Non", "Neg", "Pos"],
            title=f"Confusion Matrix (Epoch {self.current_epoch})"
        )
        wandb_logger.experiment.log({"val_cm": conf_mat_plot, "epoch": self.current_epoch})
        self.val_results.clear()

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