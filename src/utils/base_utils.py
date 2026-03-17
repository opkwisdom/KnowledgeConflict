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


def setup_logger(name: str, log_dir: str, level=logging.INFO) -> None:
    """Function to setup a logger; creates file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{name}.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.setLevel(level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('[%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s] - %(message)s')

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.WARNING)
    vllm_logger.propagate = False   # Prevent vLLM logs from propagating to the root logger
    root_logger.info(f"Logger initialized. Logs will be saved to {log_file}")

def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)