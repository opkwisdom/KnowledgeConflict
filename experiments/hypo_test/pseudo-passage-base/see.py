import pickle
from dataclasses import dataclass
from typing import List, Union, Dict

@dataclass
class TestEx:
    id: int
    question: str
    a_internal: str
    answers: List[str]
    ctx_idx: int
    ctx_rel: str
    kv_cache_score: Dict[float, int]

file_path = "kv_cache_hypothesis_test_results.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)

import pdb; pdb.set_trace()  # Set a breakpoint here to inspect 'data'