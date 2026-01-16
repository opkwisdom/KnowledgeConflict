import os
import pickle
import torch
import glob
import json
from typing import List, Dict
from dataclasses import dataclass
from pydantic import BaseModel, Field

### ==================== Data Classes ================= ###
@dataclass
class TestEx:
    id: int
    question: str
    a_internal: str
    answers: List[str]
    ctx_idx: int
    ctx_rel: str
    kv_cache_diff_score: torch.Tensor
    kv_cache_score: torch.Tensor

class MetricResult(BaseModel):
    soft_em: bool
    recall: float
    precision: float
    f1: float

class CtxsRelevance(BaseModel):
    positive: List[int] = Field(..., description="List of 0-based indices for positive contexts")
    negative: List[int] = Field(..., description="List of 0-based indices for negative contexts")
    irrelevant: List[int] = Field(..., description="List of 0-based indices for irrelevant contexts")   # Optional

    @property
    def mapping(self) -> Dict[int, str]:
        mapping = {}
        for label, idxs in [
            ("positive", self.positive),
            ("negative", self.negative),
            ("irrelevant", self.irrelevant),
        ]:
            for idx in idxs:
                mapping[idx] = label
        
        total_len = len(self.positive) + len(self.negative) + len(self.irrelevant)
        for i in range(total_len):
            if i not in mapping:
                mapping[i] = "irrelevant"
        return mapping

class InferenceResult(BaseModel):
    id: int
    question: str
    pred_answer: str
    answers: List[str]
    metrics: MetricResult
    has_answer: bool
    ctx_class: CtxsRelevance
### ==================== Data Classes ================= ###



def preprocess_inference_data(
        inference_data: Dict[str, List[InferenceResult]]
    ) -> Dict[int, CtxsRelevance]:
    processed_data = {}
    for infer_case, results in inference_data.items():
        for result in results:
            dc_result = InferenceResult(**result)
            processed_data[dc_result.id] = dc_result.ctx_class
    return processed_data


def main():
    src_data_dir = "irr_detector_data"
    target_data_dir = "irr_detector_data/labeled"   # Reference: main_20251208_060030
    output_dir = "irr_detector_data/train"
    os.makedirs(output_dir, exist_ok=True)

    all_pkls = sorted(
        glob.glob(os.path.join(src_data_dir, "*.pkl")),
        key=lambda x: int(os.path.basename(x).split("_")[-1].replace(".pkl", ""))   # 0, 1, ...
    )
    X_1, X_2, y = [], [], []

    with open(os.path.join(target_data_dir, "inference_results.json"), 'r') as f:
        target_data: Dict[str, List[InferenceResult]] = json.load(f)
    processed_target_data = preprocess_inference_data(target_data)
    
    for pkl_path in all_pkls:
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
        psg_num = list(data_dict.keys())[0]
        test_ex_list: List[TestEx] = data_dict[psg_num]

        for ex in test_ex_list:
            ctx_class = processed_target_data.get(ex.id)
            try:
                label = ctx_class.mapping[psg_num]
            except KeyError:
                label = "irrelevant"
            label = 1 if label in ["positive", "negative"] else 0
            X_1.append(ex.kv_cache_diff_score)
            X_2.append(ex.kv_cache_score)
            y.append(label)
    X_1 = torch.stack(X_1)
    X_2 = torch.stack(X_2)
    X = (X_1, X_2)
    y = torch.tensor(y)

    output_path = os.path.join(output_dir, f"kv_cache_scores_data.pkl")
    save_to_disk = {
        "kv_cache_scores": X,
        "labels": y
    }
    with open(output_path, 'wb') as f:
        pickle.dump(save_to_disk, f)
    print(f"Saved {len(X)} processed data to {output_path}")

if __name__ == "__main__":
    main()