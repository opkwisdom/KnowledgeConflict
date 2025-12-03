from tqdm import tqdm
import json

from dataclasses import asdict
from judge_model import OpenAIJudger, LLMJudger
from utils import load_relevance_dataset

def main():
    dataset = load_relevance_dataset("data/nq/parametric_relevance_tagged/validation.json")
    dataset = dataset[:10]
    checker: LLMJudger = OpenAIJudger(llm_model_name="gpt-4o-mini")

    judge_outputs = []
    for example in tqdm(dataset, desc="Judging examples"):
        judge_output = checker.judge(
            query=example.question,
            answer=example.parametric_answer,
            contexts=example.ctxs
        )
        judge_outputs.append(judge_output)
    
    print("Judging completed.")
    print(f"Total cost: ${checker.get_total_cost():.6f}")

    with open("src/test_output/judge_results.json", "w") as f:
        json.dump([output.model_dump() for output in judge_outputs], f, indent=4, ensure_ascii=False)

    with open("src/test_output/reference_results.json", "w") as f:
        json.dump([{
            "question": ex.question,
            "internal_answer": ex.parametric_answer,
            "ctx_relevance": asdict(ex.ctx_relevance)
        } for ex in dataset], f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()