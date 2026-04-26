from src.utils import load_qa_dataset
from tqdm import tqdm

dataset = load_qa_dataset("data/train/mhqa_train_half.jsonl")
error_cnt = 0
for item in tqdm(dataset):
    if item.pseudo_answer is None:
        error_cnt += 1
print(error_cnt)