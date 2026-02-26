from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
from dataclasses import dataclass, asdict
from pydantic import BaseModel
from tqdm import tqdm
import logging
import json
import os

from KVzip.model import ModelKVzip
from models import DISCA
from prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, load_relevance_dataset, compute_metrics,
    MetricResult, RelevanceQAExample
)


def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = "kvzip_test"
    config.output_dir = os.path.join(config.output_dir, experiment_name, cur_time)
    setup_logger(f"main_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_relevance_dataset(config.data.data_path)
    logger.info(f"Loaded {len(data)} examples from {config.data.data_path}")

    # Initialize model
    base_prompt = GENERATE_PROMPT[config.base_prompt_name]
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]
    logger.info(f"Using base prompt: {base_prompt}")
    logger.info(f"Using generate prompt: {generate_prompt}")

    kvzip = ModelKVzip(config.model.model_name, gen_kwargs=config.model.gen_kwargs, prompt=repeat_prompt)
    logger.info(f"Model {config.model.model_name} initialized.")
    disca = DISCA(config, kvzip, generate_prompt, base_prompt)
    logger.info("DISCA model initialized.")


if __name__ == "__main__":
    main()
