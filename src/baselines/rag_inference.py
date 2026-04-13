from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Union, Optional
import torch
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import json
import os

from src.prompt import GENERATE_PROMPT
from src.utils import (
    load_config, setup_logger, load_relevance_dataset, load_qa_dataset, has_answer, compute_metrics, MetricResult,
    apply_template,
    RelevanceQAExample, QAExample, CtxExample,
    InferenceResult,
)

# Popular cross-encoder
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"



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
    passages = [f"Title: {ctx.title}\n\n{ctx.text}" for ctx in ctxs]
    scores = reranker_model.predict([(query, passage) for passage in passages],
                                    convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False)
    _, indices = torch.sort(scores, descending=True)
    reranked_ctxs = [ctxs[idx] for idx in indices]
    return reranked_ctxs


# def construct_context(
#     ctxs: List[CtxExample],
#     relevance_map: Dict[int, str],
#     use_single_context: bool = True,
#     topk: int = -1,
# ) -> Tuple[str, str]:
#     if not ctxs:
#         return ""
#     elif use_single_context:
#         target_ctx = ctxs[0]
#         context = f"Title: {target_ctx.title}\n\n{target_ctx.text}"
#         return context, relevance_map[0]
#     else:
#         contexts = []
#         if topk > 0:
#             ctxs = ctxs[:topk]
#         else:
#             ctxs = ctxs
        
#         for ctx in ctxs:
#             contexts.append(f"Title: {ctx.title}\n\n{ctx.text}")
#         return "\n\n".join(contexts), "multiple"
    
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
        ctxs = ctxs[:topk] if topk > 0 else ctxs

        for ctx in ctxs:
            contexts.append(f"Title: {ctx.title}\n\n{ctx.text}")
        return "\n\n".join(contexts)


# def run_inference(
#     config: DictConfig,
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     data: List[RelevanceQAExample],
#     logger,
# ) -> Dict[str, List[InferenceResult]]:
#     logger.info("Starting RAG Inference (for Analysis)...")
#     inference_cases = ["param_true", "param_positive", "param_negative", "param_irrelevant", "param_multiple"]
#     results = {infer_case: [] for infer_case in inference_cases}
#     generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]

#     for idx, item in tqdm(enumerate(data), desc="Running RAG Inference", total=len(data)):
#         a_internal = item.parametric_answer
#         # is_correct = has_answer(a_internal, item.answers)
#         # Case 1 - Internal answer is correct
#         if is_correct:
#             sample_result = InferenceResult(
#                 id=idx,
#                 question=item.question,
#                 pred_answer=a_internal,
#                 answers=item.answers,
#                 is_correct=is_correct,
#             )
#             results["param_true"].append(sample_result)
#             continue
#         relevance_map = item.ctx_relevance.mapping

#         context, rel_type = construct_context(item.ctxs, relevance_map, config.data.use_single_context, topk=config.data.topk_per_query)
#         query_text = generate_prompt.format(question=item.question)
#         input_text = apply_template(query_text, context, config.model.model_name)

#         input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
#         attention_mask = torch.ones_like(input_ids).to(model.device)
#         outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, **config.model.gen_kwargs)

#         # Decode generated answer
#         gen_ids = outputs[:, input_ids.shape[1]:-1]
#         pred_answer = tokenizer.decode(gen_ids[0])

#         is_correct = has_answer(pred_answer, item.answers)
        
#         # Construct result
#         sample_result = InferenceResult(
#             id=idx,
#             question=item.question,
#             pred_answer=pred_answer,
#             answers=item.answers,
#             is_correct=is_correct,
#         )
#         results[f"param_{rel_type}"].append(sample_result)
#     return results

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
        context = construct_baseline_context(
            item.ctxs,
            config.data.use_single_context,
            topk=config.data.topk_per_query,
            do_rerank=config.data.do_rerank,
            query=item.question,
            reranker_model=reranker_model
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

def validate_and_save_results(
    inference_list: Dict[str, List[InferenceResult]],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    summary_path = f"{output_dir}/inference_summary.txt"
    all_results_path = f"{output_dir}/inference_results.json"

    total = len(inference_list)
    correct = sum([1 for res in inference_list if res.metrics.soft_em])
    recall = sum([res.metrics.recall for res in inference_list]) / total if total > 0 else 0.0
    precision = sum([res.metrics.precision for res in inference_list]) / total if total > 0 else 0.0
    f1 = sum([res.metrics.f1 for res in inference_list]) / total if total > 0 else 0.0

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Total={total}, Correct={correct}, Accuracy={accuracy:.4f},"
                f" Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved inference summary to {summary_path}")
    with open(all_results_path, 'w') as f:
        json_results = [asdict(res) for res in inference_list]
        json.dump(json_results, f, ensure_ascii=False, indent=4)


def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = f"prompt={config.generate_prompt_name}"
    output_dir = os.path.join(config.output_dir, config.data.name)  # Use data name from config
    config.output_dir = os.path.join(output_dir, experiment_name, cur_time)
    
    setup_logger(f"rag_inference_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    # if "nq" in config.data.data_path:
    #     data = load_relevance_dataset(config.data.data_path)
    # else:
    data = load_qa_dataset(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(config.model.model_name, torch_dtype="bfloat16", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Model {config.model.model_name} initialized.")

    reranker_model = CrossEncoder(RERANKER_MODEL_NAME) if config.data.do_rerank else None
    if reranker_model is not None:
        reranker_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Inference
    # if config.run_baseline:
    #     inference_results = run_baseline_inference(config, model, tokenizer, data, reranker_model, logger)
    # else:
    #     inference_results = run_inference(config, model, tokenizer, data, logger)
    inference_results = run_baseline_inference(config, model, tokenizer, data, reranker_model, logger)
    validate_and_save_results(inference_results, config.output_dir, logger)

if __name__ == "__main__":
    main()