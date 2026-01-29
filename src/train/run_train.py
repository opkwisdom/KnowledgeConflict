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

from src.train.datamodule import *
from src.train.model import *
from src.utils import load_config, setup_logger

@dataclass
class FeatureExample:
    features: torch.Tensor
    label: int

# @dataclass
# class FeatureExample:
#     features: torch.Tensor
#     h_mid: torch.Tensor
#     h_last: torch.Tensor
#     logit_diff: torch.Tensor
#     label: int

logger = logging.getLogger(__name__)

DATAMODULE_DICT = {
    "features": FeatureDataModule,
    "fusion": FusionDataModule,
}
MODEL_DICT = {
    "features": (ConflictFeatureDetector, ConflictFeatureDetectorModule),
    "fusion": (ConflictFusionDetector, ConflictFusionDetectorModule),
}

def main():
    cfg = load_config()
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Load datamodule
    datamodule = DATAMODULE_DICT[cfg.exp_type](cfg.data)

    # Load model
    model_class, module_class = MODEL_DICT[cfg.exp_type]
    model = model_class(cfg.model, cfg.model.n_class)
    lightning_module = module_class(cfg.train, model)

    # Callbacks
    output_dir = os.path.join(cfg.output_dir, cfg.exp_type)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dir,
        filename=f'{cfg.exp_type}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=3,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')

    name = f"{cfg.exp_type}_n_class={cfg.model.n_class}"
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=name,
        tags=[cfg.exp_type]
    )

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger
    )
    trainer.fit(lightning_module, datamodule=datamodule)
    # trainer.validate(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()