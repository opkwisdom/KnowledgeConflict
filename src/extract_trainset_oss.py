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
import asyncio
import numpy as np
import logging

from judge_model import CtxsRelevance, AsyncVLLMJudger
from prompt import ALL_PROMPTS, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, setup_seed, load_qa_dataset,
    MetricResult, RelevanceQAExample, QAExample
)

# NUM_SAMPLES = 10000

def get_relevance_counts(judge_outputs: List[CtxsRelevance]) -> Dict[str, int]:
    counts = {"supportive": 0, "contradictory": 0, "irrelevant": 0}
    for output in judge_outputs:
        counts["supportive"] += len(output.supportive)
        counts["contradictory"] += len(output.contradictory)
        counts["irrelevant"] += len(output.irrelevant)
    return counts


def judge_and_save(
    config: DictConfig,
    llm_judger: AsyncVLLMJudger,
    data: List[QAExample],
    save_interval: int = 2000
) -> List[RelevanceQAExample]:
    logger = logging.getLogger(__name__)
    # How many? NQ, TriviaQA equally distributed
    setup_seed(config.seed)
    n_sup, n_ctd, n_irr = 0, 0, 0
    file_name = f"full_train_data_oss.json"

    base_save_dir = os.path.join(os.path.dirname(config.data.data_path), "chunk")
    os.makedirs(base_save_dir, exist_ok=True)
    base_save_path = os.path.join(base_save_dir, file_name)

    results = []
    last_save_count = 0

    batches = [data[i:i + config.data.batch_size] for i in range(0, len(data), config.data.batch_size)]
    for i, batch in tqdm(enumerate(batches), desc="Judge Data", total=len(batches)):
        batch_queries = [item.question for item in batch]
        batch_answers = []
        for item in batch:
            ans = item.answers
            if isinstance(ans, dict):
                ans = ans.get("aliases", None)  # TriviaQA format
            batch_answers.append(ans)
        batch_ctxs = [item.ctxs for item in batch]

        # LLM Judging
        judge_output_list = asyncio.run(
            llm_judger.batch_judge(batch_queries, batch_answers, batch_ctxs)
        )
        rel_counts = get_relevance_counts(judge_output_list)
        n_sup += rel_counts["supportive"]
        n_ctd += rel_counts["contradictory"]
        n_irr += rel_counts["irrelevant"]

        rel_example = [
            RelevanceQAExample.from_qa_example(item, judge_output.mapping)
            for item, judge_output in zip(batch, judge_output_list)
        ]
        results.extend(rel_example)

        if len(results) >= save_interval:
            current_count = len(results)
            interval_end_count = last_save_count + current_count
            save_path = os.path.splitext(base_save_path)[0] + f"_{last_save_count}_{interval_end_count}.json"
            with open(save_path, 'w') as f:
                json.dump(
                    [asdict(item) for item in results],
                    f,
                    ensure_ascii=False,
                    indent=4
                )
            n_total = n_sup + n_ctd + n_irr
            logger.info(f"Total Supportive: {n_sup / n_total:.4%}, Contradictory: {n_ctd / n_total:.4%}, Irrelevant: {n_irr / n_total:.4%}")
            logger.info(f"Chunk Saved: {len(results)} items successfully written to {save_path}")
            last_save_count += current_count
            results.clear()
            n_sup, n_ctd, n_irr = 0, 0, 0
    
    if len(results) > 0:
        current_count = len(results)
        interval_end_count = last_save_count + current_count
        save_path = os.path.splitext(base_save_path)[0] + f"_{last_save_count}_{interval_end_count}.json"
        with open(save_path, 'w') as f:
            json.dump(
                [asdict(item) for item in results],
                f,
                ensure_ascii=False,
                indent=4
            )
        n_total = n_sup + n_ctd + n_irr
        logger.info(f"Total Supportive: {n_sup / n_total:.4%}, Contradictory: {n_ctd / n_total:.4%}, Irrelevant: {n_irr / n_total:.4%}")
        logger.info(f"Final Save: Total {len(results)} judged data saved to {save_path}")
        last_save_count += current_count
        results.clear()
    logger.info(f"Error Rate: {llm_judger.error_count / llm_judger.processed_count:.4%}")

    return results


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
    llm_judger = AsyncVLLMJudger(config.llm_judger)
    logger.info("GPT-OSS Judger initialized.")
    judge_and_save(config, llm_judger, data)


if __name__ == "__main__":
    main()