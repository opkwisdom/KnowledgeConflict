from check_model import OpenAIChecker
from utils import load_relevance_dataset

def main():
    dataset = load_relevance_dataset("data/nq/parametric_relevance_tagged/validation.json")
    checker = OpenAIChecker(llm_model_name="gpt-4")
    


if __name__ == "__main__":
    main()