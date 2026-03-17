from typing import List, Union, Tuple, Dict
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
from tqdm import tqdm
from dataclasses import asdict, dataclass
from vllm import LLM, SamplingParams
import json
import os
import re
import torch
import numpy as np
import logging

from models import AsyncVLLMClient
from prompt import ALL_PROMPTS, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, setup_seed, load_qa_dataset,
    MetricResult, RelevanceQAExample, QAExample
)

def augment_conflict_data(
    config: DictConfig,
    model: AsyncVLLMClient,
    sampling_params: SamplingParams,
    data: List[QAExample]
) -> None:
    logger = logging.getLogger(__name__)
    setup_seed(config.seed)
    batch_size = config.data.batch_size


def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(f"main_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load dataset
    data = load_qa_dataset(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = AsyncVLLMClient(config.model)

    # Generate augmented data
    augment_conflict_data(config, model, sampling_params, data)

if __name__ == "__main__":
    main()