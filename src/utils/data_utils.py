from dataclasses import dataclass, asdict, field, fields
from typing import Any, Dict, List, Optional, Union
import json
import os
from tqdm import tqdm
from datasets import load_dataset, DatasetDict

from .metric_utils import MetricResult

JsonType = Dict[str, Any]

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
    name: str
    pseudo_answer: Optional[str] = None
    ans_type: Optional[str] = None
    idx: int = None
    ctxs: List[CtxExample] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAExample":
        raw_ctxs = data.pop("ctxs", [])
        ctxs = [CtxExample(**ctx) for ctx in raw_ctxs]
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(ctxs=ctxs, **filtered_data)


### Append ctx relevance information to QAExample
@dataclass
class CtxsRelevance:
    supportive: List[int] = field(default_factory=list)
    contradictory: List[int] = field(default_factory=list)
    irrelevant: List[int] = field(default_factory=list)

    @property
    def mapping(self) -> Dict[int, str]:
        mapping = {}
        for label, idxs in [
            ("supportive", self.supportive),
            ("contradictory", self.contradictory),
            ("irrelevant", self.irrelevant),
        ]:
            for idx in idxs:
                mapping[idx] = label
        return mapping


@dataclass
class RelevanceQAExample(QAExample):
    ctx_relevance: CtxsRelevance = field(default_factory=CtxsRelevance)
    is_correct: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelevanceQAExample":
        data = data.copy()
        raw_ctxs = data.pop("ctxs", [])
        ctxs_obj = [CtxExample(**ctx) for ctx in raw_ctxs]
        
        raw_relevance = data.pop("ctx_relevance", {})
        # is_mapping = False
        # if raw_relevance:
        #     first_key = next(iter(raw_relevance.keys()))
        #     if isinstance(first_key, (int, str)) and str(first_key).isdigit():
        #         is_mapping = True
        
        # if is_mapping:
        #     relevance_data = {"supportive": [], "contradictory": [], "irrelevant": []}
        #     for idx, label in raw_relevance.items():
        #         relevance_data[label].append(int(idx))
        #     relevance_obj = CtxsRelevance(**relevance_data)
        # else:
        #     relevance_obj = CtxsRelevance(**raw_relevance)
        relevance_obj = CtxsRelevance(supportive=[], contradictory=[], irrelevant=[])
        
        valid_fields = {f.name for f in fields(cls)}
        init_kwargs = {k: v for k, v in data.items() if k in valid_fields}
        # init_kwargs.pop("ctx_relevance")

        return cls(ctxs=ctxs_obj, ctx_relevance=relevance_obj, **init_kwargs)

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
        for line in tqdm(f, desc="Loading QA dataset"):
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

def load_collection(data_path: str) -> List[Dict[str, str]]:
    collection = []
    with open(data_path, 'r') as f:
        for i, line in tqdm(enumerate(f), desc="Loading collection", unit=" lines"):
            if i == 0: continue  # Skip header
            try:
                id, text, title = line.strip().split("\t")
                sample = {"id": id, "text": text, "title": title}
                collection.append(sample)
            except ValueError:
                print(f"Skipping malformed line {i}: {line.strip()}")
    return collection

def format_reference_answer(answers: List[str]) -> str:
    clean_answers = [ans.replace('\xa0', ' ').strip() for ans in answers]
    
    if len(clean_answers) == 1:
        return clean_answers[0]
    elif len(clean_answers) == 2:
        return f"{clean_answers[0]} and {clean_answers[1]}"
    else:
        return ", ".join(clean_answers[:-1]) + f", and {clean_answers[-1]}"

def parse_reference_answer(formatted_answer: str) -> List[str]:
    if ", and " in formatted_answer:
        parts = formatted_answer.split(", and ")
        last_item = parts[-1]
        other_items = parts[0].split(", ")
        return other_items + [last_item]
    elif " and " in formatted_answer:
        return formatted_answer.split(" and ")
    else:
        return [formatted_answer]

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