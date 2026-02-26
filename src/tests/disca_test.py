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
from models import DISCA, SingleHiddenSCAFormer
from prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, load_relevance_dataset, compute_metrics,
    MetricResult, RelevanceQAExample
)

def test_sca_pipeline(config: DictConfig, disca: DISCA, data: List[RelevanceQAExample]):
    logger = logging.getLogger(__name__)

    batch_size = config.data.batch_size
    topk_per_query = config.data.topk_per_query

    scaformer_input_list = []
    scaformer_mask_list = []
    for start_idx in tqdm(range(len(data)), desc="Testing SCA Pipeline", total=len(data)):
        end_idx = min(start_idx + batch_size, len(data))
        batch = data[start_idx:end_idx]

        queries = [item.question for item in batch]
        contexts_list = [item.ctxs[:topk_per_query] for item in batch]
        scaformer_input, scaformer_mask = disca(queries, contexts_list)
        
        scaformer_input_list.append(scaformer_input.cpu())
        scaformer_mask_list.append(scaformer_mask.cpu())

    logger.info(f"SCA pipeline test completed. Processed {len(data)} examples in batches of {batch_size}.")
    return scaformer_input_list, scaformer_mask_list

def test_single_hidden_scaformer(
    config: DictConfig,
    scaformer: SingleHiddenSCAFormer,
    scaformer_input_list: List[torch.FloatTensor],
    scaformer_mask_list: List[torch.LongTensor]
):
    logger = logging.getLogger(__name__)
    all_outputs = []
    for scaformer_input, scaformer_mask in tqdm(zip(scaformer_input_list, scaformer_mask_list),
                                              desc="Testing SingleHiddenSCAFormer", total=len(scaformer_input_list)):
        outputs = scaformer(scaformer_input, scaformer_mask)
        all_outputs.append(outputs.cpu())
    logger.info(f"SingleHiddenSCAFormer test completed. Processed {len(scaformer_input_list)} batches.")
    return all_outputs


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
    data = data[:config.batch_size * 2]  # Only load a small subset for testing
    logger.info(f"Loaded {len(data)} examples from {config.data.data_path}")

    # Initialize model
    repeat_prompt = ALL_PROMPTS[config.self_task_prompt_name]
    base_prompt = GENERATE_PROMPT[config.base_prompt_name]
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]
    logger.info(f"Using repeat prompt: {repeat_prompt}")
    logger.info(f"Using base prompt: {base_prompt}")
    logger.info(f"Using generate prompt: {generate_prompt}")

    kvzip = ModelKVzip(config.model.model_name, gen_kwargs=config.model.gen_kwargs, prompt=repeat_prompt)
    logger.info(f"Model {config.model.model_name} initialized.")
    disca = DISCA(config, kvzip, generate_prompt, base_prompt)
    llm_tokenizer = disca.tokenizer
    logger.info("DISCA model initialized.")

    # Run test
    ### ===== First test ===== #####
    scaformer_input_list, scaformer_mask_list = test_sca_pipeline(config, disca, data)
    ### ===== Second test ===== #####
    config.scaformer.llm_width = disca.model.config.hidden_size
    scaformer = SingleHiddenSCAFormer(config.scaformer, llm_tokenizer).to(disca.device)
    scaformer_outputs = test_single_hidden_scaformer(config, scaformer, scaformer_input_list, scaformer_mask_list)

if __name__ == "__main__":
    main()
