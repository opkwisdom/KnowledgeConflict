from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, ListConfig
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoModel
import logging
import torch
import os

from models import MultiHiddenCAFormer, load_model
from datamodule import CTDataModule
from lit_modules import CTLightningModule
from utils import setup_logger, load_config


def main():
    torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf])
    
    config = load_config()
    seed_everything(config.seed)
    experiment_name = "stage1_ctr_train"
    config.output_dir = os.path.join(config.output_dir, experiment_name)
    setup_logger("main", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load datamodule & model
    datamodule = CTDataModule(config)

    llm, _ = load_model(config.model.model_name)
    config.caformer.llm_width = llm.config.hidden_size  # post-init
    caformer = MultiHiddenCAFormer(config.caformer).to(dtype=torch.bfloat16)
    roberta = AutoModel.from_pretrained("roberta-base", torch_dtype=torch.bfloat16, add_pooling_layer=False)
    
    lightning_module = CTLightningModule(config.train, llm, caformer, roberta)

    # Callbacks
    output_dir = os.path.join(config.output_dir,
                              (f"{config.exp_type}_LR={config.train.learning_rate}"
                               f"_M-W={config.train.ctr_loss.lambda_m}_freeze={config.train.freeze_pretrained}"))
    checkpoint_callback = ModelCheckpoint(
        monitor='valid/ctr_loss',
        dirpath=output_dir,
        filename=f'{config.exp_type}-{{epoch:02d}}-val_ctr_loss={{valid/ctr_loss:.4f}}',
        save_top_k=3,
        mode='min',
        every_n_train_steps=5000,
        save_last=True,
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    name = f"{config.exp_type}_LR={config.train.learning_rate}_freeze={config.train.freeze_pretrained}"
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=name,
        tags=["CT"],
        save_dir=output_dir,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        # devices=[0],
        strategy="ddp_find_unused_parameters_true",     # LLM parameters are frozen
        log_every_n_steps=10,   # More frequent logging
        max_epochs=config.train.max_epochs,
        limit_val_batches=500,    # Limit validation to 500 batches for faster validation
        val_check_interval=5000,    # Validate every 5000 training steps
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        enable_progress_bar=(not config.debug_mode),  # Debugging purpose
    )
    # trainer.fit(lightning_module, datamodule=datamodule)

    ckpt_path = None
    last_ckpt_path = os.path.join(output_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        ckpt_path = last_ckpt_path
        logger.info(f"Loading best checkpoint from {ckpt_path} for resuming...")
    else:
        logger.warning(f"No checkpoint found at {last_ckpt_path}. Skipping checkpoint loading.")
    
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
    # trainer.validate(lightning_module, datamodule=datamodule)

if __name__ == "__main__":
    main()