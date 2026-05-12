from sentence_transformers import CrossEncoder
from typing import List, Any
from tqdm import tqdm
import h5py
import logging
import json
import os
import torch
import numpy as np
from scipy.stats import spearmanr

from src.utils import load_json_data, load_h5_scores, setup_logger

logger = logging.getLogger(__name__)


RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

def rerank_contexts_by_CE(
    reranker_model: CrossEncoder,
    query: str,
    ctxs: Any
) -> List[int]:
    passages = []
    titles, sentences_list = ctxs["title"], ctxs["sentences"]
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)
        passage = f"Title: {title}\n\n{text}"
        passages.append(passage)
    scores = reranker_model.predict([(query, passage) for passage in passages],
                                    convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False)
    _, indices = torch.sort(scores, descending=True)
    return indices.tolist()

def rerank_contexts_by_oracle(
    oracle_scores: Any,
    idx: str,
    num_docs: int
) -> List[int]:
    sample_oracle_scores = oracle_scores[idx][:num_docs]  # Only consider the first num_docs passages
    _, indices = torch.sort(sample_oracle_scores, descending=True)
    return indices.tolist()

def get_gold_ctx_ids(supporting_facts, contexts) -> List[int]:
    gold_titles = supporting_facts["title"]
    titles = contexts["title"]
    gold_ids = []
    for gold_title in gold_titles:
        try:
            actual_idx = titles.index(gold_title)
            gold_ids.append(actual_idx)
        except ValueError:
            continue
    return gold_ids

# nDCG
def compute_ndcg(reranked_ctx_ids, gold_ids, klist=[1, 3, 5, 10]) -> List[float]:
    gold_ids_set = set(gold_ids)
    ndcg_scores = []
    
    for k in klist:
        dcg = 0.0
        for i, ctx_id in enumerate(reranked_ctx_ids[:k]):
            if ctx_id in gold_ids_set:
                dcg += 1 / np.log2(i + 2)  # i+2 because i starts from 0
        idcg = 0.0
        ideal_hits = min(len(gold_ids), k)
        for i in range(ideal_hits):
            idcg += 1 / np.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    return ndcg_scores

# Spearman correlation
def get_rank_positions(reranked_ids, num_docs):
    ranks = np.empty(num_docs, dtype=np.int32)
    for rank, doc_id in enumerate(reranked_ids):
        ranks[doc_id] = rank
    return ranks


def rerank_test(data, oracle_scores, reranker_model):
    ce_results = []
    oracle_results = []
    spearman_rhos = []
    spearman_pvals = []
    
    for item in tqdm(data, desc="Reranking contexts and computing NDCG"):
        idx = item["id"]
        question = item["question"]
        ctxs = item["context"]
        supporting_facts = item["supporting_facts"]
        gold_ids = get_gold_ctx_ids(supporting_facts, ctxs)

        # nDCG
        num_docs = len(ctxs['title'])
        ce_reranked_ctx_ids = rerank_contexts_by_CE(reranker_model, question, ctxs)
        oracle_reranked_ctx_ids = rerank_contexts_by_oracle(oracle_scores, idx, num_docs)
        ce_result = compute_ndcg(ce_reranked_ctx_ids, gold_ids)
        oracle_result = compute_ndcg(oracle_reranked_ctx_ids, gold_ids)
        
        # Spearman rank correlation
        ce_ranks = get_rank_positions(ce_reranked_ctx_ids, num_docs)
        oracle_ranks = get_rank_positions(oracle_reranked_ctx_ids, num_docs)
        
        if num_docs >= 2:
            rho, pval = spearmanr(ce_ranks, oracle_ranks)
            if not np.isnan(rho):
                spearman_rhos.append(rho)
                spearman_pvals.append(pval)
        
        ce_results.append(ce_result)
        oracle_results.append(oracle_result)
    ce_results = np.array(ce_results).mean(axis=0)           # (4,)
    oracle_results = np.array(oracle_results).mean(axis=0)   # (4,)
    
    spearman_mean = np.mean(spearman_rhos)
    spearman_std = np.std(spearman_rhos)
    spearman_median = np.median(spearman_rhos)
    
    correlation_summary = {
        "spearman_mean": spearman_mean,
        "spearman_std": spearman_std,
        "spearman_median": spearman_median,
        "spearman_distribution": spearman_rhos,  # for histogram
    }

    return ce_results, oracle_results, correlation_summary

def save_results(ce_results, oracle_results, correlation_summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "retrieval_vs_sys_generation_results.json")
    results = {
        "ce_results": [f"NDCG@{k}: {score}" for k, score in zip([1, 3, 5, 10], ce_results)],
        "oracle_results": [f"NDCG@{k}: {score}" for k, score in zip([1, 3, 5, 10], oracle_results)],
        "spearman": correlation_summary
    }
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # Initialize logger
    output_dir = "/workspaces/kvzip_nlplab/DISCA/src/tests/ret_vs_gen"
    setup_logger("main", output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Starting the comparison of retrieval vs generation...")

    # Load data and scores
    data_path = "/workspaces/kvzip_nlplab/data/hotpotqa-w/raw/validation_with_gold_ctx_5000.jsonl"
    # oracle_scores_path = "/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama/hotpotqa-w_val_whole_short_loss_precompute_table.h5"
    oracle_scores_path = "/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama/hotpotqa-w_sys_val_whole_short_loss_precompute_table.h5"

    data = load_json_data(data_path)
    oracle_scores = load_h5_scores(oracle_scores_path)
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME).to("cuda")

    ce_results, oracle_results, correlation_summary = rerank_test(data, oracle_scores, reranker_model)
    logger.info(f"CE Reranker NDCG Scores:")
    for k, score in zip([1, 3, 5, 10], ce_results):
        logger.info(f"NDCG@{k}: {score:.4f}")
    logger.info(f"Oracle Reranker NDCG Scores:")
    for k, score in zip([1, 3, 5, 10], oracle_results):
        logger.info(f"NDCG@{k}: {score:.4f}")
    save_results(ce_results, oracle_results, correlation_summary, output_dir)

if __name__ == "__main__":
    main()