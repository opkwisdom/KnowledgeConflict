import torch
import os
from typing import List, Union
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset
from vllm import LLM, SamplingParams
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, proj_root)

from src.prompt import GENERATE_PROMPT
from src.utils import setup_seed, setup_logger, compute_metrics, deal_judge, get_pj_prompt, QAExample

NQ_PATH = "/workspaces/kvzip_nlplab/KFC/data/nq/retrieved/train.jsonl"
TRIVIAQA_PATH = "/workspaces/kvzip_nlplab/KFC/data/triviaqa/retrieved/train.jsonl"

SEED_PATH_LIST = [
    NQ_PATH,
    TRIVIAQA_PATH,
]
SEED_VALUE = 42
BATCH_SIZE = 64

OUTPUT_PATH = "/workspaces/kvzip_nlplab/KFC/data/combined_train_data/raw_pj.jsonl"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"

def make_train_data(logger) -> Dataset:
    setup_seed(SEED_VALUE)
    total_dataset = []
    for data_path in SEED_PATH_LIST:
        dataset = load_dataset("json", data_files={"train": data_path})["train"]
        dataset = dataset.add_column("name", [data_path.split("/")[-3]] * len(dataset))
        logger.info(f"Loaded {len(dataset)} examples from {data_path}")
        
        if "triviaqa" in data_path:
            dataset = dataset.map(lambda x: {"answers": x["answers"]["aliases"]})
        shuffled_dataset = dataset.shuffle(seed=SEED_VALUE)
        total_dataset.append(shuffled_dataset)
    
    combined_dataset = concatenate_datasets(total_dataset)
    logger.info(f"Combined dataset size: {len(combined_dataset)}")
    return combined_dataset

def screen_wrong_answers(
    dataset: Dataset,
    model: LLM,
    prompt_template: str,
    logger,
    use_pj: bool = False,
) -> Dataset:
    screened_data = []
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32
    )
    num_correct = 0

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Screening Wrong Answers"):
        batch = dataset.select(range(i, min(i + BATCH_SIZE, len(dataset))))
        prompts = [
            prompt_template.format(
                question=item["question"],
            ) for item in batch
        ]
        responses = model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        
        # Priori judgment or not
        metrics = judge_prior_knowledge(batch, responses, model, use_pj, sampling_params)

        for item, response, metric in zip(batch, responses, metrics):
            generated_answer = response.outputs[0].text.strip()
            screened_data.append({
                "question": item["question"],
                "answers": item["answers"],
                "num_answer": item["num_answer"],
                "parametric_answer": generated_answer,
                "is_correct": metric,
                "name": item["name"],
                "ctxs": item["ctxs"],
            })
            num_correct += int(metric)
    
    logger.info(f"Number of correct answers: {num_correct}")
    return Dataset.from_list(screened_data)

def judge_prior_knowledge(
    batch: Dataset,
    responses,
    model: LLM,
    use_pj: bool,
    sampling_params: SamplingParams,
) -> List[bool]:
    metrics = []
    if not use_pj:
        for item, response in zip(batch, responses):
            generated_answer = response.outputs[0].text.strip()
            metric = compute_metrics(generated_answer, item["answers"])
            metrics.append(metric.soft_em)
    else:
        pj_prompts = [
            get_pj_prompt(item) for item in batch
        ]
        pj_responses = model.generate(pj_prompts, sampling_params=sampling_params, use_tqdm=False)
        for pj_response in pj_responses:
            pred = pj_response.outputs[0].text.strip().lower()
            is_giveup = deal_judge(pred)
            metrics.append(not is_giveup)  # True if the model did NOT give up

    return metrics

def main():
    logger = setup_logger("make_train_data_logger", "./logs/make_train_data")
    combined_dataset = make_train_data(logger)
    prompt = GENERATE_PROMPT["pure-llm-brief-2"]
    model = LLM(
        model=MODEL_NAME_OR_PATH,
        seed=SEED_VALUE,
        gpu_memory_utilization=0.7,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    screened_dataset = screen_wrong_answers(combined_dataset, model, prompt, logger, use_pj=True)
    screened_dataset.to_json(OUTPUT_PATH)

if __name__ == "__main__":
    main()