from transformers import AutoTokenizer
from utils import load_relevance_dataset
from tqdm import tqdm


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
data_path = "data/train/odqa_train.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_name)

THRESHOLD = [200, 210, 220, 230, 240, 250]

dataset = load_relevance_dataset(data_path)
# Iterate through the dataset and calculate document lengths
doclen_list = []
for example in tqdm(dataset):
    ctxs = example.ctxs
    for ctx in ctxs:
        ctx_text = f"Title: {ctx.title}\n\n{ctx.text}"
        tokenized_ctx = tokenizer(
            ctx_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        doclen = tokenized_ctx["input_ids"].shape[1]
        doclen_list.append(doclen)

# Calculate the percentage of documents that exceed each threshold
total_docs = len(doclen_list)
for threshold in THRESHOLD:
    count_exceeding = sum(1 for doclen in doclen_list if doclen > threshold)
    percentage_exceeding = (count_exceeding / total_docs) * 100
    print(f"Exceeding {threshold} tokens: {percentage_exceeding:.4f}%")
