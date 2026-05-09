from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Union, Optional
import torch
import logging
from peft import PeftModel, PeftConfig
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
import json
import os

from src.prompt import GENERATE_PROMPT
from src.utils import (
    load_config, setup_logger, load_relevance_dataset, load_qa_dataset, has_answer, compute_metrics, MetricResult,
    apply_template,
    validate_and_save_results,
    QAExample, CtxExample,
    InferenceResult,
)

RERANKER_TOKENIZER_NAME = 'meta-llama/Llama-2-7b-hf'
RERANKER_MODEL_NAME = 'castorini/repllama-v1-7b-lora-passage'


def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

def rerank_contexts(
    reranker_model: AutoModel,
    reranker_tokenizer: AutoTokenizer,
    query: str,
    ctxs: List[CtxExample]
) -> List[CtxExample]:
    assert reranker_model is not None, \
        "Reranker model must be provided for reranking contexts."
    assert query is not None, \
        "Query must be provided for reranking contexts."
    
    # Prepare pairs for reranking (RankLLaMA-style input)
    query_input = reranker_tokenizer(f"query: {query}</s>", return_tensors="pt").to(reranker_model.device)
    passage_texts = [f"passage: {ctx.title} {ctx.text}</s>" for ctx in ctxs]
    passage_inputs = reranker_tokenizer(
        passage_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(reranker_model.device)

    with torch.no_grad():
        # compute query embedding
        query_outputs = reranker_model(**query_input)
        query_embedding = query_outputs.last_hidden_state[0][-1]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)

        # compute passage embeddings
        passage_outputs = reranker_model(**passage_inputs)
        passage_embeddings = passage_outputs.last_hidden_state[0][-1]
        passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=0)

        # compute similarity score
        score = torch.dot(query_embedding, passage_embeddings)

    # Sort contexts by score (using ctxs)


def construct_baseline_context(
    ctxs: List[CtxExample],
    use_single_context: bool = True,
    topk: int = -1,
    do_rerank: bool = False,
    query: Optional[str] = None,
    reranker_model: AutoModel = None,
    reranker_tokenizer: AutoTokenizer = None,
) -> str:
    if not ctxs:
        return ""
    elif use_single_context:
        target_ctx = ctxs[0]
        context = f"Title: {target_ctx.title}\n\n{target_ctx.text}"
        return context
    else:
        contexts = []
        if do_rerank:
            ctxs = rerank_contexts(reranker_model, reranker_tokenizer, query, ctxs)
        else:
            pass
            # ctxs = ctxs[10:]    # Skip the first 10 passages
        ctxs = ctxs[:topk] if topk > 0 else ctxs

        for ctx in ctxs:
            contexts.append(f"Title: {ctx.title}\n\n{ctx.text}")
        return "\n\n".join(contexts)


def run_inference(
    config: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[QAExample],
    reranker_model: AutoModel,
    reranker_tokenizer: AutoTokenizer,
    logger,
) -> List[InferenceResult]:
    logger.info("Starting RankLLaMA Baseline Inference...")
    results = []
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]

    for idx, item in tqdm(enumerate(data), desc="Running RankLLaMA Inference", total=len(data)):
        context = construct_baseline_context(
            item.ctxs,
            config.data.use_single_context,
            topk=config.data.topk_per_query,
            do_rerank=config.data.do_rerank,
            query=item.question,
            reranker_model=reranker_model,
            reranker_tokenizer=reranker_tokenizer,
        )
        query_text = generate_prompt.format(question=item.question)
        input_text = apply_template(query_text, context, config.model.model_name)

        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, **config.model.gen_kwargs)
        
        # Decode generated answer
        gen_ids = outputs[:, input_ids.shape[1]:-1]
        pred_answer = tokenizer.decode(gen_ids[0])

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
    output_dir = os.path.join(config.output_dir, config.data.name)  # Use data name from config
    config.output_dir = os.path.join(output_dir, experiment_name, cur_time)
    
    setup_logger(f"rankllama_inference_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_qa_dataset(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name, torch_dtype="bfloat16", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Model {config.model.model_name} initialized.")

    reranker_model = get_model(RERANKER_MODEL_NAME)
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_TOKENIZER_NAME)
    reranker_model.to("cuda:1" if torch.cuda.is_available() else "cpu")
    
    inference_results = run_inference(config, model, tokenizer, data, reranker_model, reranker_tokenizer, logger)
    validate_and_save_results(inference_results, config.output_dir, logger)


if __name__ == "__main__":
    main()