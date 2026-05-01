from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, ListConfig
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoModel
import logging
import torch
import os
import omegaconf.base

from models import MultiHiddenCAFormerForGG, CAFormerGGClassifier, load_model
from datamodule import GGDataModule
from lit_modules import GGLightningModule, GenLossClfLightningModule
from utils import setup_logger, load_config



def load_checkpoint(model, checkpoint_path):
    logger = logging.getLogger(__name__)

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
    torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf, omegaconf.base.ContainerMetadata])
    # Allow TF32 (This can be useful for mixed precision training)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config()
    seed_everything(config.seed)
    experiment_name = "stage3_clf_only_train"
    config.output_dir = os.path.join(config.output_dir, experiment_name, "test")
    setup_logger("main", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load datamodule & model
    datamodule = GGDataModule(config)

    llm, llm_tokenizer = load_model(config.model.model_name)
    config.caformer.llm_width = llm.config.hidden_size  # post-init
    caformer = MultiHiddenCAFormerForGG(config.caformer).to(dtype=torch.bfloat16)
    # Load CAFormer weights from the best checkpoint of stage 2
    caformer, resume = load_checkpoint(caformer, config.caformer.ckpt_path)
    caformer_clf = CAFormerGGClassifier(config, caformer).to(dtype=torch.bfloat16)

    config.train.max_interleaving_len = config.data.topk_per_query * (config.data.max_seq_length + config.caformer.query_length) \
                                        + config.data.max_ans_length
    # lightning_module = GGLightningModule(config.train, llm, llm_tokenizer, caformer_clf)
    lightning_module = GenLossClfLightningModule(config.train, llm, llm_tokenizer, caformer_clf)

    # Callbacks
    from_stage2 = "fromST2" if resume else "Scratch"
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = config.data.batch_size
    output_dir = os.path.join(config.output_dir,
                              (f"{config.exp_type}_LR={config.train.learning_rate}"
                               f"_{from_stage2}_BS={batch_size}"
                               f"_ST={config.train.score_transform}"
                               f"_CMode={config.caformer.classifier_mode}"
                               f"_AMode={config.caformer.attention_mode}"
                               f"_Pool={config.caformer.pooling_strategy}"
                               f"_T={config.train.T}_Alpha={config.train.alpha}_time={current_time}"))
    checkpoint_callback = ModelCheckpoint(
        monitor='valid/loss',
        dirpath=output_dir,
        filename=f'{config.exp_type}-{{epoch:02d}}-{{step:06d}}-valid_loss={{valid/loss:.4f}}',
        save_top_k=5,
        mode='min',
        # every_n_train_steps=5000,
        save_last=True,
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    name = f"{config.exp_type}_LR={config.train.learning_rate}_BS={batch_size}_freeze={config.train.freeze_pretrained}"
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
        limit_val_batches=500,    # Limit validation to 500 batches for faster validation
        val_check_interval=0.25,    # Validate every 0.25 epochs
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        # accumulate_grad_batches=1,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        # enable_progress_bar=(not config.debug_mode),  # Debugging purpose
        inference_mode=False,
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