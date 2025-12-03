from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
import logging
import os


def load_config() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file. (YAML)")
    args, unknown = parser.parse_known_args()

    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        base_config = OmegaConf.load(args.config)
    else:
        raise ValueError("Unsupported configuration file format. Please use YAML.")
    
    cli_config = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(base_config, cli_config)
    
    return config


def setup_logger(name, log_dir: str, level=logging.INFO) -> logging.Logger:
    """Function to setup a logger; creates file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{name}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s] - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger