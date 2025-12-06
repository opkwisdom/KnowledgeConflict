from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Union, Optional
import os
import logging
from datetime import datetime

from kfc_model import KnowledgeFusionCore
from KVzip.model import ModelKVzip
from judge_model import LLMJudger, HfLLMJudger, JudgeOutput
from prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, load_relevance_dataset,
    RelevanceQAExample
)

def analysis(
    config: DictConfig,
    kfc: KnowledgeFusionCore,
    data: List[RelevanceQAExample],
    logger: logging.Logger,
):
    for idx, item in enumerate(data):
        logger.info(f"Processing item {idx+1}/{len(data)}")
        
        question = item.question
        a_internal = kfc.generate_internal_answer(question)

        pred_answer, final_rel_type, all_kv = kfc.resolve_and_generate(
            query=item.question,
            contexts=item.ctxs,
            internal_answer=a_internal,
            relevance=item.ctx_relevance,
            use_single_context=True,    # Temporary
        )
        
        for i in range(len(all_kv)):
            cache_info_output_file = os.path.join(
                config.output_dir, f"cache_info_item_{idx}_kv_{i}.json"
            )
            all_kv[i].export_cache_info(cache_info_output_file)

def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = os.path.join(config.output_dir, f"run_{cur_time}")
    os.makedirs(config.output_dir, exist_ok=True)

    logger = setup_logger("cache_analysis", config.output_dir)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_relevance_dataset(config.data.data_path)
    data = data[:10]
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Set prompts
    repeat_prompt = ALL_PROMPTS[config.self_task_prompt_name]
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]
    base_prompt = GENERATE_PROMPT["pure-llm-brief"]

    # Initialize model
    kvzip = ModelKVzip(config.model.model_name, gen_kwargs=config.model.gen_kwargs, prompt=repeat_prompt)
    logger.info(f"Model {config.model.model_name} initialized.")
    kfc = KnowledgeFusionCore(config, kvzip, generate_prompt, base_prompt, logger)
    logger.info("Knowledge Fusion Core initialized.")

    analysis(config, kfc, data, logger)

if __name__ == "__main__":
    main()