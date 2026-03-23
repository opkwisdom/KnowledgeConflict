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
from datamodule import RCDataModule
from lit_modules import RCLightningModule
from utils import setup_logger, load_config

def load_checkpoint(model, checkpoint_dir):
    logger = logging.getLogger(__name__)

    checkpoint_path = os.path.join(checkpoint_dir, "ctr_loss=4.4344.ckpt")
    if not os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint not found at {checkpoint_path}. Skipping checkpoint loading.")
        return model, False
    logger.info(f"Loading CAFormer weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cleaned_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("caformer."):
            new_key = key[len("caformer."):]
            cleaned_state_dict[new_key] = value
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    logger.info(f"Missing keys: {missing_keys}")
    logger.info(f"Unexpected keys: {unexpected_keys}")
    return model, True


def main():
    torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf])
    
    config = load_config()
    seed_everything(config.seed)
    experiment_name = "stage2_recon_train"
    config.output_dir = os.path.join(config.output_dir, experiment_name)
    setup_logger("main", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load datamodule & model
    datamodule = RCDataModule(config)

    llm, llm_tokenizer = load_model(config.model.model_name)
    config.caformer.llm_width = llm.config.hidden_size  # post-init
    caformer = MultiHiddenCAFormer(config.caformer).to(dtype=torch.bfloat16)
    # Load CAFormer weights from the best checkpoint of stage 1
    caformer, resume = load_checkpoint(caformer, config.caformer.ckpt_dir)

    lightning_module = RCLightningModule(config.train, llm, llm_tokenizer, caformer)

    # Callbacks
    from_stage1 = "fromST1" if resume else "Scratch"
    output_dir = os.path.join(config.output_dir,
                              (f"{config.exp_type}_LR={config.train.learning_rate}"
                               f"_{from_stage1}_freeze={config.train.freeze_pretrained}"))
    checkpoint_callback = ModelCheckpoint(
        monitor='valid/nll_loss',
        dirpath=output_dir,
        filename=f'{config.exp_type}-{{epoch:02d}}-{{step:06d}}-{{valid/nll_loss:.4f}}',
        save_top_k=5,
        mode='min',
        every_n_train_steps=100,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    name = f"{config.exp_type}_LR={config.train.learning_rate}_freeze={config.train.freeze_pretrained}"
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=name,
        tags=["RC"],
        save_dir=output_dir,
    )

    trainer = Trainer(
        accelerator="gpu",
        # devices="auto",
        devices=[0],
        # strategy="ddp_find_unused_parameters_true",     # LLM parameters are frozen
        log_every_n_steps=10,   # More frequent logging
        max_epochs=config.train.max_epochs,
        limit_val_batches=500,    # Limit validation to 500 batches for faster validation
        val_check_interval=100,    # Validate every 5000 training steps
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        enable_progress_bar=(not config.debug_mode),  # Debugging purpose
    )
    trainer.fit(lightning_module, datamodule=datamodule)

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