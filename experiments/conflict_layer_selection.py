from dataclasses import dataclass, asdict
from typing import List, Dict
import torch
import pickle
from pympler import asizeof
import numpy as np
import json

KEY_LIST = ["true_rel", "true_neg", "true_irr", "false_rel", "false_neg", "false_irr"]
FILEPATH = "hypo_test/base_raw/20251219_113608/kv_cache_hypothesis_test_results.pkl"
N_LAYERS = 32

@dataclass
class TestEx:
    id: int
    question: str
    a_internal: str
    answers: List[str]
    ctx_idx: int
    ctx_rel: str
    kv_cache_score: torch.Tensor

def get_real_memory_usage(data_dict):
    total_py_bytes = 0
    total_tensor_bytes = 0
    total_count = 0

    for key, examples in data_dict.items():
        total_py_bytes += asizeof.asizeof(examples)
        
        for ex in examples:
            total_count += 1
            if isinstance(ex.kv_cache_score, torch.Tensor):
                total_tensor_bytes += ex.kv_cache_score.element_size() * ex.kv_cache_score.nelement()

    total_mb = (total_py_bytes + total_tensor_bytes) / (1024 ** 2)
    print(f'[Total Memory Usage] {total_mb:.2f} MB for {total_count//len(data_dict.keys())} examples')


### Conflict layer selection via Cohen's d
def cohens_d(x, y):
    nx = x["size"]
    ny = y["size"]
    dof = nx + ny - 2
    pooled_std = ((nx - 1) * x["std"] ** 2 + (ny - 1) * y["std"] ** 2) / dof
    pooled_std = pooled_std ** 0.5
    d = (x["mean"] - y["mean"]) / pooled_std
    return d

def compute_layerwise_cohens_d(stats_a, stats_b):
    n_layers = len(stats_a)
    d_values = []
    for layer_idx in range(n_layers):
        d = cohens_d(stats_a[layer_idx], stats_b[layer_idx])
        d_values.append(d)
    return d_values

def select_conflict_layers(all_stats, target_keys, control_keys, top_k=5):
    """
    all_stats: {key: [{mean, std, size} for layer in layers]}
    """
    report = {}

    # Compute each Cohen's d
    for t_key in target_keys:
        report[t_key] = {}
        for c_key in control_keys:
            d_values = compute_layerwise_cohens_d(all_stats[t_key], all_stats[c_key])
            report[t_key][c_key] = d_values
    
    final_scores = {"max_d": [], "mean_d": [], "min_d": []}
    for layer_idx in range(N_LAYERS):
        layer_d_values = []
        for t_key in target_keys:
            for c_key in control_keys:
                layer_d_values.append(abs(report[t_key][c_key][layer_idx]))
        max_d = round(np.max(layer_d_values).item(), 6)
        mean_d = round(np.mean(layer_d_values).item(), 6)
        min_d = round(np.min(layer_d_values).item(), 6)
        final_scores["max_d"].append(max_d)
        final_scores["mean_d"].append(mean_d)
        final_scores["min_d"].append(min_d)
    
    sorted_scores = {
        "max_d": sorted(final_scores["max_d"], reverse=True),
        "mean_d": sorted(final_scores["mean_d"], reverse=True),
        "min_d": sorted(final_scores["min_d"], reverse=True),
    }
    selected_indices = np.argsort(final_scores["max_d"])[::-1].tolist()

    return {
        "selected_indices": selected_indices,
        "sorted_scores": sorted_scores,
        "final_scores": final_scores,
        "report": report,
    }


with open(FILEPATH, "rb") as f:
    results: Dict[str, List[TestEx]] = pickle.load(f)

get_real_memory_usage(results)

summary_stats = {
    _key: [
        {"size": 0, "mean": 0.0, "std": 0.0}
        for _ in range(N_LAYERS)
    ]
    for _key in KEY_LIST
}

# Compute Cohen's d
CONTROL_KEYS = ["true_irr", "false_irr"]
TARGET_KEYS = ["true_rel", "true_neg", "false_rel", "false_neg"]

for _key in KEY_LIST:
    kv_cache_scores = [ex.kv_cache_score for ex in results[_key]]
    kv_cache_tensor = torch.stack(kv_cache_scores, dim=0)   # (N, L, Top-K)

    # Get statistics per layer
    layer_means = kv_cache_tensor.mean(dim=(0, 2))  # (L,)
    layer_stds = kv_cache_tensor.std(dim=(0, 2))    # (L,)

    for layer_idx in range(kv_cache_tensor.size(1)):
        summary_stats[_key][layer_idx]["size"] = kv_cache_tensor.size(0)
        summary_stats[_key][layer_idx]["mean"] = layer_means[layer_idx].item()
        summary_stats[_key][layer_idx]["std"] = layer_stds[layer_idx].item()

    with open(f"cache_stats/kv_cache_summary_{_key}.json", "w") as f:
        json.dump(summary_stats[_key], f, indent=4)

control_stats = {
    "control_stat": [
        {
            "mean": sum([summary_stats[_key][layer_idx]["mean"] for _key in CONTROL_KEYS]) / len(CONTROL_KEYS),
            "std": sum([summary_stats[_key][layer_idx]["std"] for _key in CONTROL_KEYS]) / len(CONTROL_KEYS),
        }
        for layer_idx in range(N_LAYERS)
    ]
}
with open("cache_stats/kv_cache_control_stats.json", "w") as f:
    json.dump(control_stats, f, indent=4)

conflict_layer_info = select_conflict_layers(summary_stats, TARGET_KEYS, CONTROL_KEYS)
with open("cache_stats/conflict_layer_info.json", "w") as f:
    json.dump(conflict_layer_info, f, indent=4)