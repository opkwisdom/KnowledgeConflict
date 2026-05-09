import h5py
import numpy as np
from tqdm import tqdm
from src.utils import load_qa_dataset

MIN_GAP_LIST = [5, 4, 3]
TOPK = 50

def main():
    total_scores = []
    ds_total_scores = []
    dataset = load_qa_dataset("data/train/mhqa_train_pilot.jsonl")
    ds_ids_list = set([ds.idx for ds in dataset])

    scores_oracle_path = f"/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama/short_loss_half_precompute_table.h5"
    with h5py.File(scores_oracle_path, 'r') as f:
        for idx in tqdm(f.keys(), desc="Processing"):
            sample_scores = f[idx]["base_loss"][:] - f[idx]['doc_loss'][:]
            total_scores.append(sample_scores[10:10+TOPK].tolist())

            if idx in ds_ids_list:
                ds_total_scores.append(sample_scores[10:10+TOPK].tolist())
    total_scores = np.array(total_scores)
    ds_total_scores = np.array(ds_total_scores)

    # Analyze the distribution of score gaps
    print(total_scores.shape)  # Should be (num_samples, TOPK)
    print(ds_total_scores.shape)  # Should be (num_ds_samples, TOPK)

    top_score = np.max(total_scores, axis=1)
    bottom_score = np.min(total_scores, axis=1)
    score_gaps = top_score - bottom_score

    ds_top_score = np.max(ds_total_scores, axis=1)
    ds_bottom_score = np.min(ds_total_scores, axis=1)
    ds_score_gaps = ds_top_score - ds_bottom_score

    # Filter out samples with small score gaps
    for min_gap in MIN_GAP_LIST:
        valid_gaps = score_gaps[score_gaps >= min_gap]
        print(f"Score gap analysis for min_gap = {min_gap}:")
        print(f"Total samples: {len(score_gaps)}, Valid samples with gap >= {min_gap}: {len(valid_gaps)}")
        print(f"Valid sample percentage: {len(valid_gaps) / len(score_gaps) * 100:.2f}%")
        print(f"Average gap: {np.mean(valid_gaps):.4f}, Median gap: {np.median(valid_gaps):.4f}\n\n")
        print(f"=" * 50 + "\n\n")
        print(f"DS Score gap analysis for min_gap = {min_gap}:")
        ds_valid_gaps = ds_score_gaps[ds_score_gaps >= min_gap]
        print(f"Total DS samples: {len(ds_score_gaps)}, Valid DS samples with gap >= {min_gap}: {len(ds_valid_gaps)}")
        print(f"Valid DS sample percentage: {len(ds_valid_gaps) / len(ds_score_gaps) * 100:.2f}%")
        print(f"Average DS gap: {np.mean(ds_valid_gaps):.4f}, Median DS gap: {np.median(ds_valid_gaps):.4f}\n\n")
        print(f"=" * 50 + "\n\n")

if __name__ == "__main__":
    main()