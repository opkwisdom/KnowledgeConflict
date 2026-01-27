from typing import List, Union
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
from pydantic import BaseModel
from tqdm import tqdm
from dataclasses import asdict, dataclass
from datasets import Dataset
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

NUM_SAMPLES = 5000

@dataclass
class FeatureExample:
    features: torch.Tensor
    label: int


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

def extract_features(
    data: List[RelevanceQAExample],
    model: KnowledgeFusionCore,
) -> List[FeatureExample]:
    feature_list = []
    label_0_count = 0
    label_1_count = 0
    label_2_count = 0
    for item in tqdm(data, desc="Extract Features", total=len(data)):
        question = item.question
        answers = item.answers
        relevance = item.ctx_relevance
        if isinstance(answers, dict):
            answers = answers.get("aliases", None)  # TriviaQA format
        a_internal = item.parametric_answer
        # a_internal = model.generate_internal_answer(
        #     question,
        #     output_attentions=False,
        # )

        # Extract features
        features = model.extract_features(
            question,
            item.ctxs,
            relevance,
            a_internal,
        )
        for k, v in features.items():
            # label 0 - Non-conflict
            # label 1 - Negative conflict
            # label 2 - Positive conflict
            label = 0
            if item.is_correct:
                if k == "positive":
                    label = 0
                    label_0_count += 1
                elif k == "negative":
                    label = 1
                    label_1_count += 1
                else:
                    continue
            else:
                if k == "positive":
                    label = 2
                    label_2_count += 1
                else:
                    continue
            feature_example = FeatureExample(features=v, label=label)
            feature_list.append(feature_example)

    return feature_list, [label_0_count, label_1_count, label_2_count]


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


def load_temp_judged_data(
    config: DictConfig,
) -> List[RelevanceQAExample]:
    file_names = ["judged_data_temp_pos.json", "judged_data_temp_neg.json"]
    all_data = []
    for file_name in file_names:
        temp_load_path = os.path.join(
            os.path.dirname(config.data.data_path),
            file_name
        )
        with open(temp_load_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                all_data.append(RelevanceQAExample.from_dict(item))
    return all_data


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

    # Load temporary judged data
    judged_data = load_temp_judged_data(config)

    # Initalize model & Extract features
    repeat_prompt = ALL_PROMPTS[config.self_task_prompt_name]
    base_prompt = GENERATE_PROMPT["pure-llm-brief-2"]
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]
    kvzip = ModelKVzip(config.model.model_name, gen_kwargs=config.model.gen_kwargs, prompt=repeat_prompt)
    logger.info(f"Model {config.model.model_name} initialized.")
    kfc = KnowledgeFusionCore(config, kvzip, generate_prompt, base_prompt, logger)
    logger.info("Knowledge Fusion Core initialized.")

    features, label_counts = extract_features(judged_data, kfc)
    logger.info(f"Extracted {len(features)} feature examples.")
    logger.info(f"Label 0 - {label_counts[0]}, Label 1 - {label_counts[1]}, Label 2 - {label_counts[2]}")
    feature_save_path = os.path.join(
        os.path.dirname(config.data.data_path),
        "train_features.pt"
    )
    torch.save(features, feature_save_path)

    

if __name__ == "__main__":
    main()