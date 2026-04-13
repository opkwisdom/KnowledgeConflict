import os
import json
import random
import h5py
import torch
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import List

from utils import load_qa_dataset, CtxExample

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
    sorted, indices = torch.sort(scores, descending=True)
    reranked_ctxs = [ctxs[idx] for idx in indices]
    return reranked_ctxs, sorted

def rerank_contexts_by_table(
    idx: int,
    ctxs: List[CtxExample],
    precompute_table: dict,
    score_mode: str = "marginal-qa"
) -> List[CtxExample]:
    rerank_indices = np.argsort(precompute_table[str(idx)][score_mode][:])[::-1]
    rerank_ctxs = [ctxs[i] for i in rerank_indices]
    rerank_scores = [precompute_table[str(idx)][score_mode][i].item() for i in rerank_indices]
    return rerank_ctxs, rerank_scores



# Load dataset and results
marginal_qa_results_path = "src/results/tests/disca/nq/oracle_inference/20260413_052236/inference_results.json"
marginal_results_path = "src/results/tests/disca/nq/oracle_inference/20260413_055659/inference_results.json"

with open(marginal_results_path) as f:
    ours = json.load(f)

with open("results/rag/nq/prompt=base/20260409_052949/inference_results.json") as f:
    rag = json.load(f)

target_data = load_qa_dataset("data/nq/retrieved/validation_with_id.jsonl")


candidates = []
success_candidates = []
for i, (o, r) in enumerate(zip(ours, rag)):
    if not o["metrics"]["soft_em"] and r["metrics"]["soft_em"]:
        candidates.append(i)
    if o["metrics"]["soft_em"]:
        success_candidates.append(i)

random.seed(42)
print(f"Total candidates: {len(candidates)}")
print(f"Total success candidates: {len(success_candidates)}")
sampled_indices = random.sample(candidates, min(10, len(candidates)))
success_sampled_indices = random.sample(success_candidates, min(10, len(success_candidates)))


# Target reranking indices
nq_val_h5_path = "/workspaces/kvzip_nlplab/checkpoint/gen_precompute_table/llama/nq_val_precompute_table.h5"
precompute_table = {}
with h5py.File(nq_val_h5_path, 'r') as f:
    for sample_id in f.keys():
        precompute_table[sample_id] = {}
        for score_name in f[sample_id].keys():
            precompute_table[sample_id][score_name] = f[sample_id][score_name][:]
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder_model = CrossEncoder(RERANKER_MODEL_NAME)

TOPK = 5
total_results = []
for idx in tqdm(sampled_indices):
    item = target_data[idx]
    query = item.question
    ctxs = item.ctxs
    gold_answers = item.answers

    # rerank
    cross_reranked_ctxs, cross_scores = rerank_contexts(cross_encoder_model, query, ctxs)
    cross_reranked_ctxs, cross_scores = cross_reranked_ctxs[:TOPK], cross_scores[:TOPK]
    table_reranked_ctxs, table_scores = rerank_contexts_by_table(idx, ctxs, precompute_table, score_mode="marginal-qa")
    table_reranked_ctxs, table_scores = table_reranked_ctxs[:TOPK], table_scores[:TOPK]

    total_results.append({
        "id": idx,
        "question": query,
        "gold_answers": gold_answers,
        "cross_reranked_ctxs": cross_reranked_ctxs,
        "cross_scores": cross_scores.tolist(),
        "table_reranked_ctxs": table_reranked_ctxs,
        "table_scores": table_scores,
    })


os.makedirs("src/tests/debug/marginal", exist_ok=True)
with open("src/tests/debug/marginal/random_debug_results_10.json", "w") as f:
    json.dump(total_results, f, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)


total_results = []
for idx in tqdm(success_sampled_indices):
    item = target_data[idx]
    query = item.question
    ctxs = item.ctxs
    gold_answers = item.answers

    # rerank
    cross_reranked_ctxs, cross_scores = rerank_contexts(cross_encoder_model, query, ctxs)
    cross_reranked_ctxs, cross_scores = cross_reranked_ctxs[:TOPK], cross_scores[:TOPK]
    table_reranked_ctxs, table_scores = rerank_contexts_by_table(idx, ctxs, precompute_table, score_mode="marginal")
    table_reranked_ctxs, table_scores = table_reranked_ctxs[:TOPK], table_scores[:TOPK]

    total_results.append({
        "id": idx,
        "question": query,
        "gold_answers": gold_answers,
        "cross_reranked_ctxs": cross_reranked_ctxs,
        "cross_scores": cross_scores.tolist(),
        "table_reranked_ctxs": table_reranked_ctxs,
        "table_scores": table_scores,
    })


os.makedirs("src/tests/debug/marginal", exist_ok=True)
with open("src/tests/debug/marginal/random_debug_success_results_10.json", "w") as f:
    json.dump(total_results, f, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)


