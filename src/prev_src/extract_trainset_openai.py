from typing import List, Union, Tuple, Dict
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
from tqdm import tqdm
from dataclasses import asdict, dataclass
from datasets import Dataset
import json
import os
import re
import torch
import numpy as np
import logging

from judge_model import OpenAIJudger, JudgeOutput, CtxsRelevance
from prompt import ALL_PROMPTS, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, setup_seed, load_qa_dataset,
    MetricResult, RelevanceQAExample, QAExample
)

# NUM_SAMPLES = 10000

def get_relevance_counts(judge_outputs: CtxsRelevance) -> Dict[str, int]:
    counts = {"supportive": 0, "contradictory": 0, "irrelevant": 0}
    counts["supportive"] += len(judge_outputs.supportive)
    counts["contradictory"] += len(judge_outputs.contradictory)
    counts["irrelevant"] += len(judge_outputs.irrelevant)
    return counts


def judge_data(
    config: DictConfig,
    llm_judger: OpenAIJudger,
    data: List[QAExample],
) -> List[RelevanceQAExample]:
    logger = logging.getLogger(__name__)
    # How many? NQ, TriviaQA equally distributed
    setup_seed(config.seed)
    n_sup, n_ctd, n_irr = 0, 0, 0

    results = []
    for sample in tqdm(data, desc="Judge Data", total=len(data)):
        query = sample.question
        ans_list = sample.answers
        contexts = sample.ctxs

        # LLM Judging
        judge_output: CtxsRelevance = llm_judger.judge(
            query,
            ans_list,
            contexts,
        )
        rel_counts = get_relevance_counts(judge_output)
        n_sup += rel_counts["supportive"]
        n_ctd += rel_counts["contradictory"]
        n_irr += rel_counts["irrelevant"]

        rel_example = RelevanceQAExample.from_qa_example(sample, judge_output.mapping)
        results.append(rel_example)
    n_total = n_sup + n_ctd + n_irr
    logger.info(f"Total Supportive: {n_sup / n_total:.4%}, Contradictory: {n_ctd / n_total:.4%}, Irrelevant: {n_irr / n_total:.4%}")
    logger.info(f"Error Rate: {llm_judger.error_count / llm_judger.processed_count:.4%}")
    logger.info(f"Total Cost: {llm_judger.total_cost:.4f} USD")

    return results


def judge_and_save(
    config: DictConfig,
    llm_judger: OpenAIJudger,
    data: List[QAExample],
) -> None:
    logger = logging.getLogger(__name__)
    judged_data = judge_data(config, llm_judger, data)
    file_name = f"full_train_data_openai.json" \
        if "reasoning" not in config.llm_judger.prompt_name \
        else f"full_train_data_openai_reasoning.json"
    save_path = os.path.join(
        os.path.dirname(config.data.data_path),
        file_name
    )
    with open(save_path, 'w') as f:
        json.dump(
            [asdict(item) for item in judged_data],
            f,
            ensure_ascii=False,
            indent=4
        )
    logger.info(f"Saved {len(judged_data)} judged data to {save_path}")


def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(f"main_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_qa_dataset(config.data.data_path)
    data = data[:300]
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize judger
    llm_judger = OpenAIJudger(config.llm_judger)
    logger.info("OpenAI Judger initialized.")
    judge_and_save(config, llm_judger, data)


if __name__ == "__main__":
    main()