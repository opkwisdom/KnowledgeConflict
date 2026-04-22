import os
import torch
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def get_model_id(name: str):
    """ We support abbreviated model names such as:
        llama3.1-8b, llama3.2-*b, qwen2.5-*b, qwen3-*b, and gemma3-*b.
        The full model ID, such as "meta-llama/Llama-3.1-8B-Instruct", is also supported.
    """

    size = name.split("-")[-1].split("b")[0]  # xx-14b -> 14

    if name == "llama3.1-8b":
        return "meta-llama/Llama-3.1-8B-Instruct"
    elif name == "llama3.0-8b":
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif name == "duo":
        return "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    elif name == "llama3-8b-4m-w8a8kv4":
        return "mit-han-lab/Llama-3-8B-Instruct-Gradient-4194k-w8a8kv4-per-channel"

    elif name.startswith("llama3.2-"):
        assert size in ["1", "3"], "Model is not supported!"
        return f"meta-llama/Llama-3.2-{size}B-Instruct"

    elif name.startswith("qwen2.5-"):
        assert size in ["7", "14"], "Model is not supported!"
        return f"Qwen/Qwen2.5-{size}B-Instruct-1M"

    elif name.startswith("qwen3-"):
        assert size in ["0.6", "1.7", "4", "8", "14", "32"], "Model is not supported!"
        return f"Qwen/Qwen3-{size}B"

    elif name.startswith("gemma3-"):
        assert size in ["1", "4", "12", "27"], "Model is not supported!"
        return f"google/gemma-3-{size}b-it"

    else:
        return name  # Warning: some models might not be compatible and cause errors

def load_model(model_name: str, **kwargs):
    model_id = get_model_id(model_name)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # if do_monkey_patch:
    #     from .monkeypatch import replace_attn
    #     replace_attn(model_id)
    do_custom_attn = False
    if "custom_attn" in kwargs:
        do_custom_attn = kwargs["custom_attn"]
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map={"": local_rank},
        attn_implementation="flash_attention_2" if not do_custom_attn else "sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    if "llama" in model_id.lower():
        model.generation_config.pad_token_id = tokenizer.pad_token_id = 128004
    if "gemma-3" in model_id.lower():
        model = model.language_model
    
    logger.info(f"Model {model_id} loaded successfully.")
    return model, tokenizer