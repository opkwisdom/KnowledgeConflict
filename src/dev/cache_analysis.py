from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Union, Optional
import os

from kfc_model import KnowledgeFusionCore
from KVzip.model import ModelKVzip
from prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, load_relevance_dataset
)

def main():
    config = load_config()
    logger = setup_logger("cache_analysis", config.output_dir)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_relevance_dataset(config.data_path)
    data = data[:10]
    logger.info(f"Loaded {len(data)} data entries from {config.data_path}")

    # Set prompts
    repeat_prompt = ALL_PROMPTS[config.self_task_prompt_name]
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]
    base_prompt = GENERATE_PROMPT["pure-llm-brief"]

    # Initialize model
    kvzip = ModelKVzip(config.model_name, gen_kwargs=config.gen_kwargs, prompt=repeat_prompt)
    logger.info(f"Model {config.model_name} initialized.")
    kfc = KnowledgeFusionCore(config, kvzip, generate_prompt, base_prompt, logger)
    logger.info("Knowledge Fusion Core initialized.")

    for idx, item in enumerate(data):
        logger.info(f"Processing item {idx+1}/{len(data)}")
        context = item.ctxs
        kv = kvzip.prefill(context, load_score=False, do_score=False)
        logger.info(f"  KV cache size after prefill: {kv._mem()} GB")

        ratio = config.prune.ratio
        kv.prune(ratio=ratio)
        logger.info(f"  KV cache size after pruning (ratio={ratio}): {kv._mem()} GB")

queries = [
    "What must max_num_tokens be a multiple of when creating a cache?",
    "What bit ranges are allowed for keys and values in quantized cache layers?",
    "Which C++/CUDA file handles the implementation of dequant_cache_paged?",
]
queries = [q + "\nAnswer without explanation." for q in queries]
answers = [
    "256",
    "From 2 to 8 bits",
    "exllamav3/exllamav3_ext/cache/q_cache.cu",
]
stamp(f"Before Prefill")

kv = model.prefill(
    context,
    load_score=(args.mode == "kvzip_head"),
    do_score=(args.mode in ["kvzip", "kvzip_head"]),
)  # prefill KV cache + importance scoring
stamp(f"KV cache size: {kv._mem()} GB. After Prefill")

if args.mode in ["kvzip", "kvzip_head"]:
    ratio = 0.3 if args.mode == "kvzip" else 0.6  # compression ratio (= 1 - eviction ratio)
    kv.prune(ratio=ratio)
    stamp(f"KV cache size: {kv._mem()} GB. After Compression (ratio={ratio})")

print("-" * 100)
for i, (q, a) in enumerate(zip(queries, answers)):
    query_ids = model.apply_template(q)
    output = model.generate(query_ids, kv=kv, update_cache=False)  # efficient inference
    print(model.decode(query_ids), output, f"\n(Ground-truth: {a})")

    num_tokens = query_ids.shape[1] + model.encode(output).shape[1] + 1  # eos token
    stamp(f"After Generation", denominator=num_tokens)
    print("-" * 100)

    cache_info_output_file = f"cache_result/cache_info_{i}.json"
    if not os.path.exists("cache_result"):
        os.makedirs("cache_result", exist_ok=True)
        kv.export_cache_info(cache_info_output_file)
    