from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import sqlite3
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from omegaconf import DictConfig
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from dataclasses import asdict
import outlines
import logging
import re

from utils import CtxExample
from .template import apply_template
from .llm_judger import LLMJudger, JudgeOutput, CtxsRelevanceParser, CtxsRelevance
from .judger_prompt import HUGGINGFACE

logger = logging.getLogger(__name__)

class HfLLMJudger(LLMJudger):
    """
    HuggingFace LLM Judger implementation
    """
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.prompt = self.set_prompt(config.prompt_name)
        
        self.llm_model_name = config.llm_model_name
        self.use_cache = config.use_cache
        self.cache_path = Path(config.cache_dir) / "hf_judge_cache.db"
        self.error_count = 0
        self.processed_count = 0

        if self.use_cache:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_llm(self):
        if not self.config.use_vllm:
            tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
            tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_name,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            model.config.pad_token_id = model.config.eos_token_id
            model.eval()
            if model.generation_config is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id

            self.model = outlines.from_transformers(model, tokenizer)
        else:
            quant_type = "compressed-tensors" if getattr(self.config, "do_quant", False) else None
            self.model = LLM(
                model=self.config.llm_model_name,
                tokenizer=self.config.llm_model_name,
                dtype="bfloat16",
                quantization=quant_type,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=self.num_gpus,
                max_model_len=5120,
                disable_log_stats=True,
                seed=self.config.seed,
            )
            sp_kwargs = dict(self.config.sampling_params)
            if "stop" in sp_kwargs and sp_kwargs["stop"] is not None:
                sp_kwargs["stop"] = list(sp_kwargs["stop"])
            sp_kwargs = {k: v for k, v in sp_kwargs.items() if v is not None}
            schema_dict = CtxsRelevanceParser.model_json_schema()
            sp_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=schema_dict
            )
            self.sampling_params = SamplingParams(**sp_kwargs)

    def _init_db(self):
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT
                )
            """)
    
    def _get_cache_key(self, user_content: str) -> str:
        unique_str = f"{self.llm_model_name}|{self.prompt['system']}|{user_content}"
        return hashlib.sha256(unique_str.encode("utf-8")).hexdigest()

    def set_prompt(self, prompt: str):
        return HUGGINGFACE.get(prompt, HUGGINGFACE["base"])

    def set_template(self, prefix: str, postfix: str):
        self.prefix = prefix
        self.postfix = postfix
    
    def _prepare_input(self, query: str, ans_list: List[str], contexts: List[CtxExample]) -> str:
        joined_answer = ", ".join(ans_list)
        formatted_ctx = "\n".join([f"[{i}]\nTitle: {ctx.title}\n\n{ctx.text}\n" for i, ctx in enumerate(contexts)])
        
        user_content = self.prompt["user"].format(
            query=query,
            ref_answer=joined_answer,
            formatted_contexts=formatted_ctx,
            last_index=len(contexts)-1
        )
                
        prompt_input =  apply_template(
            query=user_content,
            context=None,
            model_name=self.config.llm_model_name,
            task="judge",
            system_prompt=self.prompt["system"],
            base_template=True,
        )
        return prompt_input
    
    def judge(self, query: str, answer: str, contexts: List[str]) -> JudgeOutput:
        # Prepare the input prompt
        formatted_ctx = "\n".join([f"[{i}]\nTitle: {ctx.title}\n\n{ctx.text}\n" for i, ctx in enumerate(contexts)])
        
        user_content = self.prompt["user"].format(
            query=query,
            internal_answer=answer,
            formatted_contexts=formatted_ctx,
            last_index=len(contexts)-1
        )
        # Check cache
        cache_key = self._get_cache_key(user_content)
        if self.use_cache:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute("SELECT data FROM cache WHERE key=?", (cache_key,))
                row = cursor.fetchone()

                if row:
                    data = json.loads(row[0])
                    cached_obj = JudgeOutput.model_validate(data)
                    return cached_obj
    
        prompt_input = apply_template(
            query=user_content,
            context=None,
            model_name=self.config.llm_model_name,
            task="judge",
            system_prompt=self.prompt["system"],
            base_template=True,
        )

        output = self.model(
            prompt_input,
            JudgeOutput,
            max_new_tokens=128,
        )
        judge_output = JudgeOutput.model_validate_json(output)
        judge_output = self.sanitize_output(judge_output, max_idx=len(contexts)-1)
        
        # Save to cache
        if self.use_cache:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, data) VALUES (?, ?)",
                    (cache_key, judge_output.model_dump_json()),
                )
        return judge_output
    
    def batch_judge(
        self,
        queries: List[str],
        answers: List[List[str]],
        contexts_list: List[List[CtxExample]]
    ) -> List[CtxsRelevance]:
        """
        batch processing utilizing vLLM
        """
        assert isinstance(self.model, LLM), \
            "`batch_judge` is only supported for vLLM-based judger."

        batch_size = len(queries)
        results: List[CtxsRelevance] = [None] * batch_size
        prompts_to_run = []
        indices_to_run = []
        cache_keys = []

        # prepare prompts and check cache
        for i, (query, ans_list, contexts) in enumerate(zip(queries, answers, contexts_list)):
            prompt_input = self._prepare_input(query, ans_list, contexts)
            cache_key = self._get_cache_key(prompt_input)

            prompts_to_run.append(prompt_input)
            cache_keys.append(cache_key)
            indices_to_run.append(i)

        # Run vLLM inference & update caches
        if prompts_to_run:
            try:
                outputs = self.model.generate(prompts_to_run, self.sampling_params, use_tqdm=False)
            except Exception as e:
                self.error_count += len(prompts_to_run)
                return []
            
            cache_updates = []
            for idx, prompt, output in zip(indices_to_run, prompts_to_run, outputs):
                indices = re.findall(r'\[(\d+)\]', prompt)
                max_idx = max((int(i) for i in indices), default=-1)
                raw_text = output.outputs[0].text
                try:
                    parsed_obj = CtxsRelevanceParser.model_validate_json(raw_text)
                    final_obj = parsed_obj.to_dataclass()
                except Exception as e:
                    self.error_count += 1
                    final_obj = CtxsRelevance()
                self.processed_count += max_idx + 1

                results[idx] = final_obj
                if self.use_cache:
                    cache_updates.append((cache_keys[idx], json.dumps(asdict(final_obj))))

            # Insert new cache entries in batch
            if self.use_cache and cache_updates:
                with sqlite3.connect(self.cache_path) as conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO cache (key, data) VALUES (?, ?)",
                        cache_updates
                    )

        return results