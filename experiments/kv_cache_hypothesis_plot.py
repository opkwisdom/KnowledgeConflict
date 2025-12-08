import pickle
from dataclasses import dataclass
from typing import List, Union, Dict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from datetime import datetime
from omegaconf import OmegaConf

from utils import load_config, setup_logger, load_json_data
from KVzip.model import ModelKVzip


@dataclass
class TestEx:
    id: int
    question: str
    a_internal: str
    answers: List[str]
    ctx_idx: int
    ctx_rel: str
    kv_cache_score: Union[torch.Tensor, Dict[float, int]]

ScoreRatio = Dict[float, float]

BINS = 20

def load_pickled_results(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_kv_cache_hypothesis(
    candidate_1: ScoreRatio,
    candidate_2: ScoreRatio,
    baseline_1: ScoreRatio,
    baseline_2: ScoreRatio,
    title: str,
    output_path: str,
    num_bins: int = BINS
) -> None:
    figure, ax = plt.subplots(figsize=(12, 8))

    # Set X axis & width
    bin_edges = np.linspace(0, 1, num_bins)
    bar_x_positions = bin_edges[:-1]
    bar_width = bin_edges[1] - bin_edges[0]

    center_x_positions = bin_edges[:-1] + bar_width / 2
    individual_bar_width = bar_width / 5

    offset_left = -2 * individual_bar_width
    offset_right = 2 * individual_bar_width

    sorted_keys = sorted(candidate_1.keys())

    # Extract values
    candidate_1_values = np.array([candidate_1[k] for k in sorted_keys])
    candidate_2_values = np.array([candidate_2[k] for k in sorted_keys])
    baseline_1_values = np.array([baseline_1[k] for k in sorted_keys])
    baseline_2_values = np.array([baseline_2[k] for k in sorted_keys])

    # Compute X positions
    x_pos_cand1 = center_x_positions + offset_left
    x_pos_cand2 = center_x_positions + offset_left + individual_bar_width
    x_pos_base1 = center_x_positions
    x_pos_base2 = center_x_positions + individual_bar_width

    # Plot bars
    ax.bar(x_pos_cand1, candidate_1_values, width=individual_bar_width, label='false_neg', color='blue', alpha=0.8)
    ax.bar(x_pos_cand2, candidate_2_values, width=individual_bar_width, label='false_rel', color='orange', alpha=0.8)
    ax.bar(x_pos_base1, baseline_1_values, width=individual_bar_width, label='true_irr', color='green', alpha=0.8)
    ax.bar(x_pos_base2, baseline_2_values, width=individual_bar_width, label='false_irr', color='red', alpha=0.8)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    # Set x-ticks (0.1 intervals)
    major_xticks = np.arange(0, 1.1, 0.1)
    ax.set_xticks(major_xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in major_xticks])
    ax.set_xlim(0, 1)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path)
    plt.close(figure)


def plot_kv_cache_hypothesis_raw_by_layer(
    candidates: Dict[str, torch.Tensor],
    baseline_1: torch.Tensor,
    baseline_2: torch.Tensor,
    title: str,
    output_path: str,
    layer_idx: int,
    use_log_scale: bool = False,
) -> None:
    candidate_keys = ["true_rel", "false_rel", "true_neg", "false_neg"]
    figure, axes = plt.subplots(figsize=(16, 16), nrows=2, ncols=2)
    axes: plt.Axes = axes.flatten()

    def plot_each_ax(
        ax: plt.Axes,
        base1_layer_scores: np.ndarray,
        base2_layer_scores: np.ndarray,
        cand_layer_scores: np.ndarray,
        cand_key: str,
        num_bins: int = 10
    ):
        topk = base1_layer_scores.shape[0]
        sample_indices = np.linspace(0, topk - 1, num_bins+1, dtype=int)
        x_sampled = sample_indices + 1

        x = np.arange(1, topk+1)

        y_min = min(
            base1_layer_scores.min(),
            base2_layer_scores.min(),
            cand_layer_scores.min()
        ) - 1

        base1_scores_sampled = base1_layer_scores[sample_indices]
        base2_scores_sampled = base2_layer_scores[sample_indices]
        cand_scores_sampled = cand_layer_scores[sample_indices]

        base1_layer_scores_gap = base1_scores_sampled - y_min
        base2_layer_scores_gap = base2_scores_sampled - y_min
        cand_layer_scores_gap = cand_scores_sampled - y_min

        if use_log_scale:
            base1_layer_scores_gap = np.log(base1_layer_scores_gap)
            base2_layer_scores_gap = np.log(base2_layer_scores_gap)
            cand_layer_scores_gap = np.log(cand_layer_scores_gap)

        bar_width = (x_sampled[1] - x_sampled[0]) / 4

        ax.bar(x_sampled - bar_width, cand_layer_scores_gap, width=bar_width, label=cand_key, color="green", alpha=0.6)
        ax.bar(x_sampled, base2_layer_scores_gap, width=bar_width, label="false_irr", color="orange", alpha=0.6)
        ax.bar(x_sampled + bar_width, base1_layer_scores_gap, width=bar_width, label="true_irr", color="blue", alpha=0.6)

        # ax.plot(x, base1_layer_scores_gap, label="true_irr", color="blue", linewidth=2)
        # ax.fill_between(x, base1_layer_scores_gap, alpha=0.3, color="blue")
        # ax.plot(x, base2_layer_scores_gap, label="false_irr", color="orange", linewidth=2)
        # ax.fill_between(x, base2_layer_scores_gap, alpha=0.3, color="orange")
        # ax.plot(x, cand_layer_scores_gap, label=cand_key, color="green", linewidth=2)
        # ax.fill_between(x, cand_layer_scores_gap, alpha=0.3, color="green")

        ax.set_xticks(x_sampled)
        x_label = "Top-K"
        y_label = "Score Gap" if not use_log_scale else "Score Gap (log scale)"
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(cand_key, fontsize=14)

        ax.set_xlim(x_sampled.min() - bar_width*2, x_sampled.max() + bar_width*2)

        ax.legend(fontsize=10)


    for i, cand_key in enumerate(candidate_keys):
        ax = axes[i]
        candidate = candidates[cand_key]

        # Extract layer scores
        cand_layer_scores = candidate[:, layer_idx, :].numpy().mean(axis=0)
        base1_layer_scores = baseline_1[:, layer_idx, :].numpy().mean(axis=0)
        base2_layer_scores = baseline_2[:, layer_idx, :].numpy().mean(axis=0)

        plot_each_ax(
            ax,
            base1_layer_scores,
            base2_layer_scores,
            cand_layer_scores,
            cand_key,
        )
    figure.suptitle(title, fontsize=18)
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(figure)


def extract_kv_cache_score_ratio(data: List[TestEx], logger) -> ScoreRatio:
    total_bins = [i / BINS for i in range(1, BINS)]
    bin_ratio_dict = {i: 0 for i in total_bins}

    for item in data:
        for bin_key in total_bins:
            bin_ratio_dict[bin_key] += item.kv_cache_score[bin_key]

    n_items = sum(bin_ratio_dict.values())
    for bin_key in total_bins:
        bin_ratio_dict[bin_key] /= n_items

    logger.info(f"KV Cache Score Ratio: {bin_ratio_dict}")
    return bin_ratio_dict


def main():
    # Load configuration
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if config.calc_type is not None:
        subdir = f"{config.prompt_name}_{config.calc_type}"
    else:
        subdir = f"{config.prompt_name}"
    subdir = f"{subdir}_log" if config.use_log_scale else subdir

    config.output_dir = os.path.join(config.output_dir, subdir)     # for different prompts
    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logger(f"kv_cache_hypothesis_plot_{cur_time}", config.output_dir)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_pickled_results(config.data.data_path)
    logger.info(f"Loaded {len(data)} data entries from {config.data.data_path}")

    # Plotting
    if config.calc_type != "raw":
        baseline_1 = extract_kv_cache_score_ratio(data["true_irr"], logger)
        baseline_2 = extract_kv_cache_score_ratio(data["false_irr"], logger)

        candidate_keys = [
            ["true_neg", "true_rel"],
            ["false_neg", "false_rel"]
        ]
        for cand_key in candidate_keys:
            candidate_1 = extract_kv_cache_score_ratio(data[cand_key[0]], logger)
            candidate_2 = extract_kv_cache_score_ratio(data[cand_key[1]], logger)
            output_path = os.path.join(config.output_dir, f"kv_cache_hypothesis_{cand_key[0]}_{cand_key[1]}_plot.png")
            title = f"KV Cache Hypothesis Test - {cand_key[0]} & {cand_key[1]}"
            plot_kv_cache_hypothesis(candidate_1, candidate_2, baseline_1, baseline_2, title, output_path)
    else:
        baseline_1 = torch.stack(
            [torch.tensor(item.kv_cache_score) for item in data["true_irr"]],
            dim=0
        )   # (n_samples, n_layers, topk)
        baseline_2 = torch.stack(
            [torch.tensor(item.kv_cache_score) for item in data["false_irr"]],
            dim=0
        )

        candidates = {}
        candidate_keys = ["true_rel", "false_rel", "true_neg", "false_neg"]
        for cand_key in candidate_keys:
            candidates[cand_key] = torch.stack(
                [torch.tensor(item.kv_cache_score) for item in data[cand_key]],
                dim=0
            )
        n_layers = baseline_1.size(1)
        output_dir = os.path.join(config.output_dir, str(config.topk))
        os.makedirs(output_dir, exist_ok=True)

        for layer_idx in tqdm(range(n_layers), desc="Plotting by layer"):
            if config.use_log_scale:
                output_file = f"{layer_idx}_log.png"
            else:
                output_file = f"{layer_idx}.png"

            output_path = os.path.join(output_dir, output_file)
            title = f"Attention Weights - Layer {layer_idx}"
            plot_kv_cache_hypothesis_raw_by_layer(
                candidates,
                baseline_1,
                baseline_2,
                title,
                output_path,
                layer_idx,
                use_log_scale=config.use_log_scale,
            )


if __name__ == "__main__":
    main()