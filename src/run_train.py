from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import logging
import torch
import os

from models import DISCA, SingleHiddenCAFormer, MultiHiddenCAFormer, CAFormerClassifier
from datamodule import CADataModule
from lit_modules import CAFormerLightningModule
from utils import setup_logger, load_config


def main():
    config = load_config()
    seed_everything(config.seed)
    experiment_name = "train_caformer"
    config.output_dir = os.path.join(config.output_dir, experiment_name)
    setup_logger("main", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load datamodule & model
    datamodule = CADataModule(config.data)
    disca = DISCA(config, config.model.model_name)
    config.caformer.llm_width = disca.model.config.hidden_size  # post-init
    caformer = SingleHiddenCAFormer(config.caformer, disca.tokenizer).to(dtype=torch.bfloat16)
    caformer_clf = CAFormerClassifier(config, caformer).to(dtype=torch.bfloat16)
    lightning_module = CAFormerLightningModule(config.train, disca, caformer_clf)

    # Callbacks
    output_dir = os.path.join(config.output_dir,
                              f"{config.exp_type}_LR={config.train.learning_rate}_CTR-W={config.train.ctr_loss_weight}_freeze={config.train.freeze_pretrained}")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dir,
        filename=f'{config.exp_type}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=3,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    name = f"{config.exp_type}_LR={config.train.learning_rate}_CTR-W={config.train.ctr_loss_weight}_freeze={config.train.freeze_pretrained}"
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=name,
        save_dir=output_dir,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        # devices=[0],
        strategy="ddp_find_unused_parameters_true",     # LLM parameters are frozen
        log_every_n_steps=10,   # More frequent logging
        max_epochs=config.train.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        # enable_progress_bar=(not config.debug_mode),  # Debugging purpose
    )
    trainer.fit(lightning_module, datamodule=datamodule)
    # trainer.validate(lightning_module, datamodule=datamodule)

if __name__ == "__main__":
    main()