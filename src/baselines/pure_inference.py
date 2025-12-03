from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import logging
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from dataclasses import dataclass

from utils import setup_logger, load_config, load_json_data

def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))