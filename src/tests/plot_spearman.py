import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import sys
import pandas as pd

def plot_spearman(data, output_path):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    data = np.array(data["spearman"]["spearman_distribution"])
    n_samples = len(data)
    
    sns.histplot(data, kde=True, color="#3498db", bins=50)
    plt.title(f"Spearman Correlation Distribution ({n_samples})", pad=15, fontweight="bold")
    plt.ylabel("Density", labelpad=10)
    plt.xlabel("Spearman Correlation")

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    # test_type, do_vllm = sys.argv[1], bool(sys.argv[2])
    input_path = f"results/ret_vs_gen/sys_retrieval_vs_generation_results.json"
    output_path = f"results/ret_vs_gen/sys_spearman_plot.png"

    with open(input_path, "r") as f:
        data = json.load(f)
    plot_spearman(data, output_path)
    
if __name__ == "__main__":
    main()