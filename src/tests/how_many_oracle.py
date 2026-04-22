import json

PATH_LIST = [
    "data/hotpotqa-w/raw/train_p.jsonl",
    "data/hotpotqa-w/raw/validation_p.jsonl",
    "data/hotpotqa-w/raw/test_p.jsonl",
    "data/2wiki/raw/train_p.jsonl",
    "data/2wiki/raw/validation_p.jsonl",
    "data/2wiki/raw/test_p.jsonl",
]

def report_oracle_counts(path: str):
    oracle_count = 0
    total_count = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            item = json.loads(line)
            if item.get("pseudo_answer") is not None:
                oracle_count += 1
    print(f"{path}: {oracle_count} / {total_count} ({(oracle_count/total_count)*100:.2f}%)")


def main():
    for path in PATH_LIST:
        report_oracle_counts(path)

if __name__ == "__main__":
    main()