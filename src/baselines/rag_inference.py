from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from typing import List, Union, Optional
import torch
import logging
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import os

from src.prompt import GENERATE_PROMPT
from src.utils import (
    load_config, setup_logger, load_qa_dataset, compute_metrics, validate_and_save_results,
    apply_template,
    QAExample, CtxExample,
    InferenceResult,
)



def rerank_contexts(
    reranker_model: CrossEncoder,
    query: str,
    ctxs: List[CtxExample],
):
    assert reranker_model is not None, \
        "Reranker model must be provided for reranking contexts."
    assert query is not None, \
        "Query must be provided for reranking contexts."
    
    # Prepare pairs for reranking
    ctxs = ctxs[10:60]    # Skip the first 10 passages, only consider the retrieved ones
    passages = [f"Title: {ctx.title}\n\n{ctx.text}" for ctx in ctxs]
    scores = reranker_model.predict([(query, passage) for passage in passages],
                                    convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False)
    _, indices = torch.sort(scores, descending=True)
    reranked_ctxs = [ctxs[idx] for idx in indices]
    return reranked_ctxs
    
def construct_baseline_context(
    ctxs: List[CtxExample],
    use_single_context: bool = True,
    topk: int = -1,
    do_rerank: bool = False,
    query: Optional[str] = None,
    reranker_model: CrossEncoder = None,
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
            ctxs = rerank_contexts(reranker_model, query, ctxs)
        else:
            ctxs = ctxs[10:]    # Skip the first 10 passages
        ctxs = ctxs[:topk] if topk > 0 else ctxs

        for ctx in ctxs:
            contexts.append(f"Title: {ctx.title}\n\n{ctx.text}")
        return "\n\n".join(contexts)

def run_baseline_inference(
    config: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[QAExample],
    reranker_model: Union[CrossEncoder, None],
    logger,
) -> List[InferenceResult]:
    logger.info("Starting RAG Baseline Inference...")
    results = []
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]

    for idx, item in tqdm(enumerate(data), desc="Running RAG Inference", total=len(data)):
        # Compare
        # if item.pseudo_answer is None:
        #     continue

        context = construct_baseline_context(
            item.ctxs,
            config.data.use_single_context,
            topk=config.data.topk_per_query,
            do_rerank=config.do_rerank,
            query=item.question,
            reranker_model=reranker_model
        )
        query_text = generate_prompt.format(question=item.question)
        input_text = apply_template(query_text, context, config.model.model_name)

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
    if config.data.do_rerank:
        reranker_model_name = getattr(config, "reranker_model_name", None)
        experiment_name += f"_{reranker_model_name.split('/')[-1]}"

    config.output_dir = os.path.join(output_dir, experiment_name)
    
    setup_logger(f"rag_inference_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_qa_dataset(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(config.model.model_name, torch_dtype="bfloat16", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Model {config.model.model_name} initialized.")

    reranker_model_name = getattr(config, "reranker_model_name", None)
    if config.data.do_rerank and reranker_model_name is None:
        logger.warning("Reranking is enabled but no reranker model name provided. Reranking will be skipped.")
        reranker_model = None
    else:
        reranker_model = CrossEncoder(reranker_model_name)
        reranker_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Reranker model {reranker_model_name} initialized.")
    
    inference_results = run_baseline_inference(config, model, tokenizer, data, reranker_model, logger)
    validate_and_save_results(inference_results, config.output_dir, logger)

if __name__ == "__main__":
    main()