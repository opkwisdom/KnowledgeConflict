from typing import List, Union, Tuple
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
from pydantic import BaseModel
from tqdm import tqdm
from dataclasses import asdict, dataclass
from datasets import Dataset
from collections import Counter
import json
import os
import torch
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(current_dir, "../../"))
kvzip_path = os.path.join(proj_root, "KVzip")
sys.path.insert(0, proj_root)
sys.path.insert(0, kvzip_path)

from KVzip.model import ModelKVzip
from src.kfc_model import KnowledgeFusionCore
from src.judge_model import OpenAIJudger, JudgeOutput, CtxsRelevance
from src.prompt import ALL_PROMPTS, GENERATE_PROMPT
from src.utils import (
    setup_logger, load_config, setup_seed, load_qa_dataset,
    MetricResult, RelevanceQAExample, QAExample
)

NUM_SAMPLES = 10000

@dataclass
class FeatureExample:
    features: torch.Tensor
    h_mid: torch.Tensor
    h_last: torch.Tensor
    logit_diff: torch.Tensor
    label: int


def _determine_label(context_type: str) -> int:
    # label 0 - Positive
    # label 1 - Negative
    # label 2 - Irrelevant
    if context_type == "positive":
        return 0
    elif context_type == "negative":
        return 1
    else:
        return 2


def judge_data(
    config: DictConfig,
    llm_judger: OpenAIJudger,
    data: List[QAExample],
    is_correct_filter: bool = True,
) -> List[RelevanceQAExample]:
    # How many? NQ, TriviaQA equally distributed
    setup_seed(42)
    ds = [item for item in data if item.is_correct == is_correct_filter]
    indices = np.random.randint(0, len(ds), NUM_SAMPLES)
    sampled = [ds[i] for i in indices]

    results = []
    for idx, item in tqdm(enumerate(sampled), desc="Judge Data", total=len(sampled)):
        question = item.question
        answers = item.answers
        if isinstance(answers, dict):
            answers = answers.get("aliases", None)  # TriviaQA format
        a_internal = item.parametric_answer
        
        # LLM Judging
        judge_output: CtxsRelevance = llm_judger.judge(
            question,
            a_internal,
            item.ctxs,
        )
        rel_example = RelevanceQAExample.from_qa_example(item, judge_output.mapping)
        results.append(rel_example)
        
    return results


def judge_and_save(
    config: DictConfig,
    llm_judger: OpenAIJudger,
    data: List[QAExample],
    is_correct_filter: bool = True,
    logger = None
) -> None:
    judged_data = judge_data(config, llm_judger, data, is_correct_filter=is_correct_filter)
    file_name = "judged_data_temp_pos.json" if is_correct_filter else "judged_data_temp_neg.json"
    temp_save_path = os.path.join(
        os.path.dirname(config.data.data_path),
        file_name
    )
    with open(temp_save_path, 'w') as f:
        for item in judged_data:
            json_line = json.dumps(asdict(item), ensure_ascii=False)
            f.write(json_line + "\n")
    if logger:
        logger.info(f"Saved {len(judged_data)} judged data to {temp_save_path}")


def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(f"main_{cur_time}", config.output_dir)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_qa_dataset(config.data.data_path)
    # data = data[:500]
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize judger
    llm_judger = OpenAIJudger(config.judger)
    logger.info("LLM Judger initialized.")
    judge_and_save(config, llm_judger, data, is_correct_filter=True, logger=logger)
    judge_and_save(config, llm_judger, data, is_correct_filter=False, logger=logger)


if __name__ == "__main__":
    main()