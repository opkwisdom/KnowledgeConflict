import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

TOPK_RANGE = 10

def plot_dist(f, mode: str = "whole"):
    total_scores = []
    for idx in tqdm(f.keys(), desc=f"Processing {mode}"):
        if mode in f[idx]:
            sample_scores = f[idx][mode][:]
            total_scores.extend(sample_scores.tolist())
    
    print(f"Total scores collected for {mode}: {len(total_scores)}")
    
    total_scores = np.array(total_scores)

    # remove outliers for better visualization (0.5% ~ 99.5%)
    p_low, p_high = np.percentile(total_scores, [0.5, 99.5])
    filtered_scores = total_scores[(total_scores >= p_low) & (total_scores <= p_high)]
    print(f"Scores after removing outliers for {mode}: {len(filtered_scores)}, range: [{p_low:.4f}, {p_high:.4f}]")
    
    # Two kinds of graphs
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Distribution - {mode} (Log Scale Y)")
    plt.hist(total_scores, bins=100, density=True, alpha=0.6, color="blue")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.title(f"Zoomed In - {mode} (Middle 99%)")
    plt.hist(filtered_scores, bins=100, density=True, alpha=0.6, color="orange")
    
    plt.tight_layout()
    plt.savefig(f"src/tests/oracle_dist/scores_oracle_dist_{mode}.png")
    plt.close()

def main():
    scores_oracle_path = "/workspaces/kvzip_nlplab/checkpoint/gen_precompute_table/llama/precompute_table.h5"
    os.makedirs("src/tests/oracle_dist", exist_ok=True)
    with h5py.File(scores_oracle_path, "r") as f:
        plot_dist(f, mode="whole")
        plot_dist(f, mode="marginal")
        for i in range(TOPK_RANGE):
            plot_dist(f, mode=f"topk_{i+1}")

if __name__ == "__main__":
    main()