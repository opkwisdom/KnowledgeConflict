import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import sys
import pandas as pd

def plot_genloss(data_list, output_path):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    n_samples = len(data_list)
    rows = []
    for item in data_list:
        if "base_loss" in item and item["base_loss"]:
            rows.append({"Context Type": "Base (No Context)", "Loss": item["base_loss"][0]})
        
        if "gold_loss" in item and item["gold_loss"]:
            for g_loss in item["gold_loss"]:
                rows.append({"Context Type": "Gold (True Target)", "Loss": g_loss})
            
        if "neg_loss" in item and item["neg_loss"]:
            for n_loss in item["neg_loss"]:
                rows.append({"Context Type": "Neg (Distractor)", "Loss": n_loss})

    df = pd.DataFrame(rows)
    category_order = [
        "Base (No Context)", 
        "Neg (Distractor)", 
        "Gold (True Target)"
    ]
    ax = sns.boxplot(
        x="Context Type",
        y="Loss",
        data=df,
        width=0.5,
        order=category_order,
        palette=["#aeb6bf", "#2ecc71", "#e74c3c"],
        showfliers=True,
        fliersize=3
    )
    plt.title(f"Generation Loss Comparison by Context Type ({n_samples})", pad=15, fontweight="bold")
    plt.ylabel("Generation Loss (Lower is Better)", labelpad=10)
    plt.xlabel("")

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
def plot_genloss_diff(data_list, output_path):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    n_samples = len(data_list)
    rows = []
    for item in data_list:
        base_loss = 0.0
        if "base_loss" in item and item["base_loss"]:
            base_loss = item["base_loss"][0]
        
        if "gold_loss" in item and item["gold_loss"]:
            for g_loss in item["gold_loss"]:
                rows.append({"Context Type": "Gold (True Target)", "Loss": g_loss - base_loss})
            
        if "neg_loss" in item and item["neg_loss"]:
            for n_loss in item["neg_loss"]:
                rows.append({"Context Type": "Neg (Distractor)", "Loss": n_loss - base_loss})
    
    df = pd.DataFrame(rows)
    category_order = [
        "Neg (Distractor)", 
        "Gold (True Target)"
    ]
    ax = sns.boxplot(
        x="Context Type",
        y="Loss",
        data=df,
        width=0.5,
        order=category_order,
        palette=["#e74c3c", "#2ecc71"],
        showfliers=True,
        fliersize=3
    )
    plt.title(f"Generation Margin Loss Comparison by Context Type ({n_samples})", pad=15, fontweight="bold")
    plt.ylabel("Generation Margin Loss (Lower is Better)", labelpad=10)
    plt.xlabel("")

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    

def calculate_and_save_ranking_stats(data_list, output_meta_path):
    gold_ranks, base_ranks = [], []
    gold_is_first_count = 0
    total_valid_queries = 0

    for item in data_list:
        if not (item.get("gold_loss") and item.get("base_loss") and item.get("neg_loss")):
            continue
        gold_losses = item["gold_loss"]
        neg_losses = item["neg_loss"]
        base_loss = item["base_loss"][0]

        all_losses = [("gold", l) for l in gold_losses] + [("base", base_loss)] + [("neg", l) for l in neg_losses]
        all_losses.sort(key=lambda x: x[1])  # Sort by loss (lower is better)

        gold_rank = next(i for i, v in enumerate(all_losses) if v[0] == "gold") + 1
        base_rank = next(i for i, v in enumerate(all_losses) if v[0] == "base") + 1

        gold_ranks.append(gold_rank)
        base_ranks.append(base_rank)

        if gold_rank == 1:
            gold_is_first_count += 1

        total_valid_queries += 1

    avg_gold_rank = np.mean(gold_ranks)
    std_gold_rank = np.std(gold_ranks)
    avg_base_rank = np.mean(base_ranks)
    std_base_rank = np.std(base_ranks)
    total_docs_per_query = len(all_losses)
    flip_ratio = (gold_is_first_count / total_valid_queries) * 100

    # 텍스트 파일로 저장
    with open(output_meta_path, "w", encoding="utf-8") as f:
        f.write("=== Generation Loss Ranking Statistics ===\n")
        f.write(f"Total Queries Evaluated: {total_valid_queries}\n")
        f.write(f"Total Contexts per Query: {total_docs_per_query} (1 Gold, 1 Base, {total_docs_per_query-2} Negs)\n\n")
        f.write(f"1. Gold Rank 1 Ratio (Flip Ratio): {flip_ratio:.2f}%\n")
        f.write(f"   -> Gold Context가 가장 낮은 Loss를 기록한 쿼리의 비율\n\n")
        
        gold_lower_bound = max(1, avg_gold_rank - std_gold_rank)
        gold_upper_bound = min(total_docs_per_query, avg_gold_rank + std_gold_rank)
        f.write(f"2. Average Gold Rank: {avg_gold_rank:.2f} / {total_docs_per_query}\n")
        f.write(f"   -> 1-Std Range: [{gold_lower_bound:.2f}, {gold_upper_bound:.2f}]\n\n")

        base_lower_bound = max(1, avg_base_rank - std_base_rank)
        base_upper_bound = min(total_docs_per_query, avg_base_rank + std_base_rank)
        f.write(f"3. Average Base Rank: {avg_base_rank:.2f} / {total_docs_per_query}\n")
        f.write(f"   -> 1-Std Range: [{base_lower_bound:.2f}, {base_upper_bound:.2f}]\n\n")
    return flip_ratio, avg_gold_rank, avg_base_rank


def main():
    test_type, do_vllm = sys.argv[1], bool(sys.argv[2])
    input_path = f"src/tests/oracle_loss/{test_type}_genloss_test_result.json" \
        if not do_vllm else f"src/tests/oracle_loss/{test_type}_vllm_genloss_test_result.json"
    output_path = f"src/tests/oracle_loss/{test_type}_genloss_plot.png" \
        if not do_vllm else f"src/tests/oracle_loss/{test_type}_vllm_genloss_plot.png"
    output_meta_path = f"src/tests/oracle_loss/{test_type}_ranking_stats.txt" \
        if not do_vllm else f"src/tests/oracle_loss/{test_type}_vllm_ranking_stats.txt"
    output_margin_path = f"src/tests/oracle_loss/{test_type}_margin_plot.png" \
        if not do_vllm else f"src/tests/oracle_loss/{test_type}_vllm_margin_plot.png"

    # with open(input_path, "r") as f:
    #     data = json.load(f)

    plot_genloss(data, output_path)
    plot_genloss_diff(data, output_margin_path)
    calculate_and_save_ranking_stats(data, output_meta_path)

if __name__ == "__main__":
    main()