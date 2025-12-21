from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Union
import json
import os
from tqdm import tqdm
from datasets import load_dataset, DatasetDict

from .metric_utils import MetricResult


### Base QA Example Dataclasses
@dataclass
class CtxExample:
    hasanswer: bool
    id: int
    score: float
    text: str
    title: str


@dataclass
class QAExample:
    question: str
    answers: List[str]
    num_answer: int
    parametric_answer: Optional[str] = None
    ctxs: List[CtxExample] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAExample":
        raw_ctxs = data.pop("ctxs", [])
        ctxs = [CtxExample(**ctx) for ctx in raw_ctxs]
        return cls(ctxs=ctxs, **data)


### Append ctx relevance information to QAExample
@dataclass
class CtxsRelevance:
    positive: List[int] = field(default_factory=list)
    negative: List[int] = field(default_factory=list)
    irrelevant: List[int] = field(default_factory=list)

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
        return mapping


@dataclass
class RelevanceQAExample(QAExample):
    ctx_relevance: CtxsRelevance = field(default_factory=CtxsRelevance)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelevanceQAExample":
        data = data.copy()
        raw_ctxs = data.pop("ctxs", [])
        ctxs_obj = [CtxExample(**ctx) for ctx in raw_ctxs]
        raw_relevance = data.pop("ctx_relevance", {})
        relevance_obj = CtxsRelevance(**raw_relevance)
        return cls(ctxs=ctxs_obj, ctx_relevance=relevance_obj, **data)

    @classmethod
    def from_qa_example(cls, qa_example: QAExample, ctx_relevance: CtxsRelevance) -> "RelevanceQAExample":
        return cls(ctx_relevance=ctx_relevance, **asdict(qa_example))


### Result Dataclasses
@dataclass
class InferenceResult:
    id: int
    question: str
    pred_answer: str
    answers: List[str]
    metrics: MetricResult



def load_qa_dataset(data_path: str) -> List[QAExample]:
    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            dataset.append(QAExample.from_dict(item))
    return dataset

def load_json_data(data_path: str):
    print(f"Loading data from {data_path}...")
    ext = os.path.splitext(data_path)[1]

    if ext == '.jsonl':
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif ext == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def load_relevance_dataset(data_path: str) -> List[RelevanceQAExample]:
    raw_data = load_json_data(data_path)

    if not isinstance(raw_data, list):
        raise ValueError("Loaded JSON data must be a list of QA examples.")
    
    relevance_examples = [
        RelevanceQAExample.from_dict(item) 
        for item in tqdm(raw_data, desc="Parsing QA Examples")
    ]
    
    print(f"Successfully loaded {len(relevance_examples)} Relevance QA examples.")
    return relevance_examples


if __name__ == "__main__":
    # Example usage
    NQ_DIR = "../../data/nq/retrieved"
    train_path = os.path.join(NQ_DIR, "train.jsonl")
    validation_path = os.path.join(NQ_DIR, "validation.jsonl")
    dataset = load_dataset("json", data_files={"train": train_path, "validation": validation_path})
    train = dataset["train"]

    for item in train:
        import pdb; pdb.set_trace()
        qa_example = QAExample.from_dict(item)