from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import logging
import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, proj_root)

from src.train.datamodule import ConflictDataModule
from src.train.model import ConflictDetector, ConflictDetectorModule
from src.utils import load_config, setup_logger

@dataclass
class FeatureExample:
    features: torch.Tensor
    label: int

logger = logging.getLogger(__name__)

def main():
    cfg = load_config()
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Load datamodule
    datamodule = ConflictDataModule(cfg.data)

    # Load model
    model = ConflictDetector(cfg.model)
    lightning_module = ConflictDetectorModule(cfg.train, model)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.output_dir,
        filename='conflict-detector-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=WandbLogger(project=cfg.project_name),
    )
    trainer.fit(lightning_module, datamodule=datamodule)
    # trainer.validate(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()