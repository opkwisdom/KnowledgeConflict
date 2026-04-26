import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

def plot_dist(f):
    total_scores = []
    for idx in tqdm(f.keys(), desc="Processing"):
        sample_scores = f[idx]["base_loss"][:] - f[idx]['doc_loss'][:]
        total_scores.extend(sample_scores.tolist())

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
    plt.savefig(f"src/tests/oracle_dist/scores_genloss_oracle_dist.png")
    plt.close()

def main():
    scores_oracle_path = f"/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama/loss_half_precompute_table.h5"
    os.makedirs("src/tests/oracle_dist", exist_ok=True)
    with h5py.File(scores_oracle_path, "r") as f:
        plot_dist(f)
        # plot_dist(f, mode="marginal")
        # for i in range(TOPK_RANGE):
        #     plot_dist(f, mode=f"topk_{i+1}")

if __name__ == "__main__":
    main()