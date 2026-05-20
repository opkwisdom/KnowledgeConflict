import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
from tqdm import tqdm
from src.utils import load_qa_dataset

def plot_dist(f):
    total_scores = []
    for idx in tqdm(f.keys(), desc="Processing"):
        sample_scores = f[idx]["base_loss"][:] - f[idx]['doc_loss'][:]
        # n_positive = np.sum(sample_scores[10:10+50] > 0)
        # if n_positive == 0:
        #     continue
        # total_scores.extend(sample_scores.tolist())
        total_scores.extend(sample_scores[10:10+50].tolist())

    print(f"Total scores collected: {len(total_scores)}")

    total_scores = np.array(total_scores)

    # remove outliers for better visualization (0.5% ~ 99.5%)
    p_low, p_high = np.percentile(total_scores, [0.5, 99.5])
    filtered_scores = total_scores[(total_scores >= p_low) & (total_scores <= p_high)]
    print(f"Scores after removing outliers for: {len(filtered_scores)}, range: [{p_low:.4f}, {p_high:.4f}]")
    
    # Two kinds of graphs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Distribution - (Log Scale Y)")
    plt.hist(total_scores, bins=100, density=True, alpha=0.6, color="blue")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.title(f"Zoomed In - (Middle 99%)")
    plt.hist(filtered_scores, bins=100, density=True, alpha=0.6, color="orange")
    
    plt.tight_layout()
    plt.savefig(f"src/tests/oracle_dist/sys_short_wo_gold_genloss_oracle_dist.png")
    plt.close()

def ds_dependent_plot_dist(f, ds_ids_list, name):
    total_ds_ids_list = set(ds_ids_list)
    total_scores = []
    for idx in tqdm(f.keys(), desc="Processing"):
        if idx not in total_ds_ids_list:
            continue
        sample_scores = f[idx]["base_loss"][:] - f[idx]['doc_loss'][:]
        # total_scores.extend(sample_scores.tolist())
        total_scores.extend(sample_scores[10:10+50].tolist())

    print(f"Total scores collected: {len(total_scores)}")

    total_scores = np.array(total_scores)

    # remove outliers for better visualization (0.5% ~ 99.5%)
    p_low, p_high = np.percentile(total_scores, [0.5, 99.5])
    filtered_scores = total_scores[(total_scores >= p_low) & (total_scores <= p_high)]
    print(f"Scores after removing outliers for: {len(filtered_scores)}, range: [{p_low:.4f}, {p_high:.4f}]")
    
    # Two kinds of graphs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Distribution - (Log Scale Y)")
    plt.hist(total_scores, bins=100, density=True, alpha=0.6, color="blue")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.title(f"Zoomed In - (Middle 99%)")
    plt.hist(filtered_scores, bins=100, density=True, alpha=0.6, color="orange")
    
    plt.tight_layout()
    plt.savefig(f"src/tests/oracle_dist/{name}_sys_short_wo_gold_genloss_oracle_dist.png")
    plt.close()
    
    
def summarize_positive_ratio(f):
    total_scores = []
    for idx in tqdm(f.keys(), desc="Processing"):
        sample_scores = f[idx]["base_loss"][:] - f[idx]['doc_loss'][:]
        # total_scores.extend(sample_scores.tolist())
        n_positive = np.sum(sample_scores[10:10+50] > 0)
        if n_positive == 0:
            continue
        total_scores.append(sample_scores[10:10+50].tolist())
    total_scores = np.array(total_scores)
    print(f"Total scores collected: {total_scores.shape}")

    total_scores = np.array(total_scores)
    
    # Get positive ratio summary
    def compute_positive_ratio(scores):
        N, TOPK = scores.shape
        bins = np.arange(TOPK+1)
        positive_counts = np.sum(scores > 0, axis=1)
        positive_bins = np.sum(positive_counts[:, None] >= bins, axis=0)
        
        post_positive_bins = np.concatenate([positive_bins[1:], np.array([0])])
        marginal_positive_bins = positive_bins - post_positive_bins
        return (marginal_positive_bins / N).tolist()
    
    total_positive_ratio_summary = compute_positive_ratio(total_scores)
    print(sum(total_positive_ratio_summary))
    output_path = "src/tests/oracle_dist/sys_short_positive_wo_gold_genloss_oracle_dist_summary.json"
    outputs = {i: ratio for i, ratio in enumerate(total_positive_ratio_summary)}
    with open(output_path, 'w') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)


def main():
    # dataset = load_qa_dataset("data/train/mhqa_train_pilot.jsonl")
    # ds_ids_list = [ds.idx for ds in dataset]

    scores_oracle_path = f"/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama/sys_short_loss_half_precompute_table.h5"
    # scores_oracle_path = f"/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama/short_loss_half_precompute_table.h5"
    os.makedirs("src/tests/oracle_dist", exist_ok=True)
    with h5py.File(scores_oracle_path, "r") as f:
        plot_dist(f)
        # summarize_positive_ratio(f)
        # ds_dependent_plot_dist(f, ds_ids_list, name="pilot")
        # plot_dist(f, mode="marginal")
        # for i in range(TOPK_RANGE):
        #     plot_dist(f, mode=f"topk_{i+1}")

if __name__ == "__main__":
    main()