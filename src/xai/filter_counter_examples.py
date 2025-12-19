from tqdm import tqdm
import re
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass
import os

from .base_utils import setup_logger
from .metric_utils import em_for_interpretability

DATASET_PATH = "../../data/nq-swap"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

BASE_USER_TEMPLATE = "Question: {question}\n\n"
RAG_USER_TEMPLATE = "Context: {context}\n\nQuestion: {question}\n\n"

MODEL_KWARGS = {
    "max_new_tokens": 32,
    "do_sample": False,
    "top_p": 1.0,
}
BATCH_SIZE = 16

def preprocess_function(examples):
    pattern = r"</?[a-zA-Z0-9_]+>"  # remove HTML tags
    targets = ["org_context", "sub_context"]
    
    for col in targets:
        examples[col] = [re.sub(pattern, "", text) for text in examples[col]]
    return examples


def generate_batch(model, tokenizer, messages_list, model_kwargs):
    texts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_list
    ]

    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        padding_side="left" 
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            **model_kwargs
        )

    generated_texts = []
    input_len = inputs.input_ids.shape[1]
    
    for output in outputs:
        generated_text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        generated_texts.append(generated_text)
        
    return generated_texts


def main():
    logger = setup_logger("filter_counter_examples", "xai/logs/filter_counter_examples.log")
    dataset = load_from_disk(DATASET_PATH)["dev"]
    dataset = dataset.map(preprocess_function, batched=True)
    logger.info(f"Loaded dataset from {DATASET_PATH} with {len(dataset)} examples.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()
    MODEL_KWARGS["pad_token_id"] = tokenizer.eos_token_id

    filtered_examples = []

    data_list = list(dataset)

    for i in tqdm(range(0, len(data_list), BATCH_SIZE), desc="Processing batches"):
        batch = data_list[i:i+BATCH_SIZE]

        # Step 1: Knowledge check without context
        messages_list = []
        for ex in batch:
            user_content = BASE_USER_TEMPLATE.format(question=ex["question"])
            messages_list.append([{"role": "user", "content": user_content}])
        no_context_responses = generate_batch(model, tokenizer, messages_list, MODEL_KWARGS)

        survivors_step1 = []
        for ex, resp in zip(batch, no_context_responses):
            if em_for_interpretability(resp, ex["org_answer"]):
                survivors_step1.append(ex)
        
        if not survivors_step1:
            continue

        # Step 2: Conflict check with substituted context
        messages_list = []
        for ex in survivors_step1:
            user_content = RAG_USER_TEMPLATE.format(context=ex["sub_context"], question=ex["question"])
            messages_list.append([{"role": "user", "content": user_content}])
        sub_context_responses = generate_batch(model, tokenizer, messages_list, MODEL_KWARGS)

        survivors_step2 = []
        for ex, resp in zip(survivors_step1, sub_context_responses):
            # only keep examples where the model answers the substituted answer
            if em_for_interpretability(resp, ex["sub_answer"]) and not em_for_interpretability(resp, ex["org_answer"]):
                survivors_step2.append(ex)
        
        if not survivors_step2:
            continue
        
        for ex in survivors_step2:
            filtered_examples.append(ex)

    filtered_dataset = Dataset.from_list(filtered_examples)
    filtered_dataset.save_to_disk("../data/nq-swap-filtered")

if __name__ == "__main__":
    main()