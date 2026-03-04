from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import logging
import os

from datamodule import CADataModule
from lit_modules import CAFormerLightningModule
from utils import setup_logger, load_config


def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = "train_caformer"
    config.output_dir = os.path.join(config.output_dir, experiment_name, cur_time)
    setup_logger(f"main_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load datamodule & model
    datamodule = CADataModule(config.data)
    lightning_module = CAFormerLightningModule(config.train, disca, caformer)


if __name__ == "__main__":
    main()