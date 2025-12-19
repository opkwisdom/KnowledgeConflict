import pickle
from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class LogitLensOutputs:
    probs_target: List[float]
    probs_counter: List[float]
    input_type: str
    is_divergent_at_first: bool
    topk_values: torch.Tensor   # (Layers, TopK), Topk probs for each layer
    topk_indices: torch.Tensor # (Layers, TopK), Topk indices for each layer
    target_idx: int
    counter_idx: int

with open("logit_lens_results.pkl", "rb") as f:
    logit_lens_results: List[LogitLensOutputs] = pickle.load(f)

import pdb; pdb.set_trace()

