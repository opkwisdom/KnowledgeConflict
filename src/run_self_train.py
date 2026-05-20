import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, ListConfig
from datetime import datetime
import logging

import os
import typing
import omegaconf.base
import collections

from models import load_model
from datamodule import GGDataModule, GGItDataModule
from lit_modules import GenLossSelfLightningModule
from utils import setup_logger, load_config


def main():
    torch.serialization.add_safe_globals([
        DictConfig, ListConfig, OmegaConf, omegaconf.base.ContainerMetadata,
        typing.Any, dict, collections.defaultdict
    ])
    # Allow TF32 (This can be useful for mixed precision training)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config()
    seed_everything(config.seed)
    experiment_name = "self_train"
    config.output_dir = os.path.join(config.output_dir, experiment_name)
    setup_logger("main", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load datamodule & model
    datamodule = GGDataModule(config) if not config.use_it else GGItDataModule(config)
    logger.info(f"DataModule {datamodule.__class__.__name__} initialized.")

    llm, llm_tokenizer = load_model(config.model.model_name)
    llm_tokenizer.padding_side = getattr(config.model, "padding_side", "right")
    lightning_module = GenLossSelfLightningModule(config.train, llm, llm_tokenizer)

    # Callbacks
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = config.data.batch_size * config.train.accumulate_grad_batches
    output_dir = os.path.join(config.output_dir,
                              (f"{config.exp_type}_LR={config.train.learning_rate}"
                               f"_Scratch_BS={batch_size}"
                               f"_Epochs={config.train.max_epochs}"
                               f"_Pool={config.train.pooling_strategy}"
                               f"_T={config.train.T}_Alpha={config.train.alpha}_SYS={config.use_it}"))
    best_checkpoint_callback = ModelCheckpoint(
        monitor='valid/loss',
        dirpath=output_dir,
        filename=f'{config.exp_type}-{{epoch:02d}}-{{step:06d}}-valid_loss={{valid/loss:.4f}}',
        save_top_k=-1,
        mode='min',
        save_last=True,
        auto_insert_metric_name=False
    )
    
    world_size = torch.cuda.device_count()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    name = f"Self-Train_{config.exp_type}_LR={config.train.learning_rate}_BS={batch_size}"
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=name,
        tags=["GG_CLF"],
        save_dir=output_dir,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        # devices=[0],
        # strategy="ddp",     # DDP without find_unused_parameters since CAFormer is frozen
        strategy="ddp_find_unused_parameters_true",     # LLM parameters are frozen
        log_every_n_steps=5,   # More frequent logging
        max_epochs=config.train.max_epochs,
        limit_val_batches=int(500 / world_size),    # Limit validation to 500 batches for faster validation
        val_check_interval=0.25,    # Validate every 0.25 epochs
        gradient_clip_val=2.0,
        callbacks=[best_checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        # accumulate_grad_batches=1,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        precision="bf16-mixed",
        # enable_progress_bar=False,  # Debugging purpose
        inference_mode=True,
    )

    ckpt_path = None
    last_ckpt_path = os.path.join(output_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        ckpt_path = last_ckpt_path
        logger.info(f"Loading best checkpoint from {ckpt_path} for resuming...")
    else:
        logger.warning(f"No checkpoint found at {last_ckpt_path}. Skipping checkpoint loading.")
    
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
    # trainer.validate(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()