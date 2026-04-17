import os
from argparse import ArgumentParser
from dataclasses import asdict
from typing import List
from tqdm import tqdm
from openai import OpenAI
import json

from utils import (
    load_qa_dataset,
    QAExample,
    CtxsRelevance,
    RelevanceQAExample
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_MODEL = "gpt-4o-mini"


# SYSTEM_PROMPT = (
#     "You are an extremely strict judge evaluating RAG (Retrieval-Augmented Generation) systems. "
#     "Your task is to classify the relationship between a Question, a Answer, and a Context. "
#     "You must verify if the Context ACTUALLY contains the specific information required by the Answer.\n\n"

#     "Classify into one of three categories based on these strict rules:\n\n"
    
#     "1. Relevant: The Context contains the information necessary to derive the Answer. "
#     "Deduction or general knowledge is NOT allowed. The evidence must be present in the text.\n"
    
#     "2. Negative: The Context discusses the same entity, topic, or keywords as the Question, but it FAILS to provide the Answer. "
#     "This is a 'hard negative' or 'distractor'.\n"
    
#     "3. Irrelevant: The Context is unrelated to the Question, discusses a different entity, or is completely off-topic.\n\n"
    
#     "Output ONLY one word: 'Relevant', 'Negative', or 'Irrelevant'."
# )

SYSTEM_PROMPT = (
    "You are an extremely strict judge evaluating RAG (Retrieval-Augmented Generation) systems. "
    "Your task is to classify the relationship between a Question, a Answer, and a Context. "
    "You must verify if the Context ACTUALLY contains the specific information required by the Answer.\n\n"

    "Classify into one of three categories based on these strict rules:\n\n"
    
    "1. Relevant: The Context contains the specific information to derive the Answer literally. "
    "Deduction or general knowledge is NOT allowed. The evidence must be present in the text.\n"
    
    "2. Negative: The Context discusses the same entity, topic, or keywords as the Question, but it FAILS to provide the Answer. "
    "This is a 'hard negative' or 'distractor'.\n"
    
    "3. Irrelevant: The Context is unrelated to the Question, discusses a different entity, or is completely off-topic.\n\n"
    
    "Output ONLY one word: 'Relevant', 'Negative', or 'Irrelevant'."
)

# SYSTEM_PROMPT = (
#     "You are an extremely strict judge evaluating RAG (Retrieval-Augmented Generation) systems. "
#     "Your task is to classify the relationship between a Question, a Answer, and a Context. "
#     "You must verify if the Context ACTUALLY contains the specific information required by the Answer.\n\n"

#     "Classify into one of three categories based on these strict rules:\n\n"
    
#     "1. Relevant: The Context contains the specific information to derive the Answer literally. "
#     "Deduction or general knowledge is NOT allowed. The evidence must be present in the text.\n"
    
#     "2. Negative: The Context discusses the same entity, topic, or keywords as the Question, but it FAILS to provide the Answer. "
#     "This is a 'hard negative' or 'distractor'.\n"
    
#     "3. Irrelevant: The Context is unrelated to the Question, discusses a different entity, or is completely off-topic.\n\n"

#     "Output ONLY one word: 'Relevant', 'Negative', or 'Irrelevant'."
# )

def api_call(model, question: str, answer: str, context: str) -> bool:
    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Context: {context}\n\n"
        f"Classify the context (Relevant/Negative/Irrelevant):"
    )
    try:
        response = model.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0,
            timeout=30
        )
        reply = response.choices[0].message.content.strip().lower()

        if "relevant" in reply and "irrelevant" not in reply:
            return "relevant"
        elif "negative" in reply:
            return "negative"
        else:
            return "irrelevant"
    except Exception as e:
        print(f"API call failed: {e}")
        return "error"

def tag_relevance(client, dataset_path, output_path):
    # Load dataset
    data: List[QAExample] = load_qa_dataset(dataset_path)
    
    all_results: List[RelevanceQAExample] = []

    # Relevance tagging API call
    print("Tagging relevance using LLM...")
    for i, ex in tqdm(enumerate(data), total=len(data), desc="Processing examples"):
        relevance = CtxsRelevance()

        question = ex.question
        answer = ex.answers
        ctxs = [f"Title: {ctx.title}\n\nText: {ctx.text}" for ctx in ex.ctxs]

        ctxs_correct = [ctx.hasanswer for ctx in ex.ctxs]
        for idx, ctx in enumerate(ctxs):
            llm_rel = api_call(client, question, answer, ctx)
            if llm_rel == "relevant":
                relevance.positive.append(idx)
            elif llm_rel == "negative":
                relevance.negative.append(idx)
            elif llm_rel == "irrelevant":
                relevance.irrelevant.append(idx)
            else:
                print(f"Skipping context {idx} due to API error.")
            # if ctxs_correct[idx]:
            #     relevance.positive.append(idx)
            #     continue
            # else:
            #     # Prepare prompt
            #     llm_rel = api_call(client, question, answer, ctx)
            #     # Judge relevance
            #     if llm_rel and not ctxs_correct[idx]:
            #         relevance.negative.append(idx)
            #     else:
            #         relevance.irrelevant.append(idx)

        all_results.append(RelevanceQAExample.from_qa_example(ex, relevance))

    # Save tagged results
    output_path = output_path.replace(".jsonl", ".json")
    with open(output_path, 'w') as f:
        json.dump([asdict(ex) for ex in all_results], f, indent=4)


def load_config():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/nq/retrieved")
    args = parser.parse_args()
    return args

def main():
    # Extract configurations
    cfg = load_config()
    output_dir = cfg.data_dir.replace('retrieved', "relevance_tagged")
    os.makedirs(output_dir, exist_ok=True)

    data_file = os.path.join(cfg.data_dir, 'validation.jsonl')

    output_file = os.path.join(output_dir, os.path.basename(data_file))
    tag_relevance(
        client,
        dataset_path=data_file,
        output_path=output_file
    )
    # for data_file in data_files:
    #     output_file = os.path.join(output_dir, os.path.basename(data_file))
    #     tag_relevance(
    #         model,
    #         dataset_path=data_file,
    #         output_path=output_file
    #     )

if __name__ == "__main__":
    main()