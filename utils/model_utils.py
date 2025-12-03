from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model_name(name: str):
    """ We support abbreviated model names such as:
        llama3.1-8b, llama3.2-*b, qwen2.5-*b, qwen3-*b, and gemma3-*b.
        The full model ID, such as "meta-llama/Llama-3.1-8B-Instruct", is also supported.
    """

    if name == "meta-llama/Llama-3.1-8B-Instruct":
        return "llama3.1-8b"
    elif name == "meta-llama/Meta-Llama-3-8B-Instruct":
        return "llama3.0-8b"
    elif name == "Qwen/Qwen2.5-7B-Instruct-1M":
        return "qwen2.5-7b"
    elif name == "google/gemma-3-12b-it":
        return "gemma3-12b"
    else:
        raise ValueError("Model name not supported!")


def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation='flash_attention_2'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer