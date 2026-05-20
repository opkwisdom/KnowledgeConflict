from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Union, Optional
import torch
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
import json
import os

from src.prompt import GENERATE_PROMPT
from src.utils import (
    load_config, setup_logger, load_qa_dataset, compute_metrics, has_answer, validate_and_save_results,
    apply_template,
    RelevanceQAExample, CtxExample, QAExample,
    InferenceResult,
)

def run_baseline_inference(
    config: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[QAExample],
    logger,
) -> List[InferenceResult]:
    logger.info("Starting Pure Baseline Inference...")
    results = []
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]

    for idx, item in tqdm(enumerate(data), desc="Running Pure Inference", total=len(data)):
        query_text = generate_prompt.format(question=item.question)
        input_text = apply_template(query_text, None, config.model.model_name, base_template=True)
        input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False).to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, **config.model.gen_kwargs)

        # Decode generated answer
        input_len = input_ids.shape[1]
        gen_ids = outputs[:, input_len:]
        pred_answer = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        answers = item.answers
        if isinstance(answers, dict):
            answers = answers.get("aliases", None)  # TriviaQA format

        metrics = compute_metrics(pred_answer, answers)
        
        # Construct result
        sample_result = InferenceResult(
            id=idx,
            question=item.question,
            pred_answer=pred_answer,
            answers=answers,
            metrics=metrics,
        )
        results.append(sample_result)
    return results


def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = f"prompt={config.generate_prompt_name}"
    output_dir = os.path.join(config.output_dir, config.model.model_name.split('/')[-1], config.data.name)  # Use data name from config
    config.output_dir = os.path.join(output_dir, experiment_name)

    setup_logger(f"pure_inference_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_qa_dataset(config.data.data_path)
    # data = data[:50]
    
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(config.model.model_name, torch_dtype="bfloat16", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Model {config.model.model_name} initialized.")

    # Inference
    inference_results = run_baseline_inference(config, model, tokenizer, data, logger)
    validate_and_save_results(inference_results, config.output_dir, logger)

if __name__ == "__main__":
    main()