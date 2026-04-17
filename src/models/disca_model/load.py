import os
import torch
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch.nn.attention.flex_attention as pt_fa_module
import transformers.integrations.flex_attention as hf_fa_module

logger = logging.getLogger(__name__)

torch._dynamo.config.cache_size_limit = 64

def _patch_flex_attention_num_stages():
    """
    Fix for Ampere GPUs (e.g. A6000): Triton's default num_stages=4 generates
    kernels requiring ~107KB shared memory, but A6000's per-block limit is ~99KB.
    Patching flex_attention_compiled in transformers to use num_stages=2.

    Python looks up module globals at call time, so replacing the module-level
    `flex_attention_compiled` is sufficient — no need to touch compile_friendly_flex_attention.
    """
    try:
        _orig_flex_attention = pt_fa_module.flex_attention
        def _patched_pt_flex_attention(*args, **kwargs):
            kwargs.pop("training", None)
            opts = kwargs.get('kernel_options') or {}
            opts["num_stages"] = 2
            opts["BLOCK_M"] = 64
            opts["BLOCK_N"] = 64
            kwargs['kernel_options'] = opts

            return _orig_flex_attention(*args, **kwargs)
        
        pt_fa_module.flex_attention = torch.compile(
            _patched_pt_flex_attention, 
            dynamic=False
        )
        # pt_fa_module.flex_attention = _patched_pt_flex_attention
        # hf_fa_module.flex_attention_compiled = torch.compile(_patched_pt_flex_attention, dynamic=True)
        if hasattr(hf_fa_module, "compile_friendly_flex_attention"):
            hf_fa_module.compile_friendly_flex_attention = pt_fa_module.flex_attention
        logger.info("flex_attention patched: num_stages=2 (A6000 shared-memory fix)")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not patch flex_attention_compiled: {e}")

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

    do_flex = kwargs.get("do_flex", False)
    if do_flex:
        _patch_flex_attention_num_stages()
        pass

    # if do_monkey_patch:
    #     from .monkeypatch import replace_attn
    #     replace_attn(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map={"": local_rank},
        attn_implementation="flash_attention_2" if not do_flex else "flex_attention"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    if "llama" in model_id.lower():
        model.generation_config.pad_token_id = tokenizer.pad_token_id = 128004
    if "gemma-3" in model_id.lower():
        model = model.language_model
    
    # Optimization
    # if do_flex:
    #     logger.info("Compiling patched model with torch.compile...")
    #     model = torch.compile(model, dynamic=True)
    
    logger.info(f"Model {model_id} loaded successfully.")
    return model, tokenizer