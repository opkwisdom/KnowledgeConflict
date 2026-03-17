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
from models import AsyncVLLMClient
from .template import apply_template
from .llm_judger import LLMJudger, JudgeOutput, CtxsRelevanceParser, CtxsRelevance
from .judger_prompt import HUGGINGFACE

logger = logging.getLogger(__name__)


class AsyncVLLMJudger(LLMJudger):
    """
    Asynchronous vLLM Judger implementation
    """
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.prompt = self.set_prompt(config.prompt_name)
        
        self.llm_model_name = config.llm_model_name
        self.use_cache = config.use_cache
        self.cache_path = Path(config.cache_dir) / "vllm_judge_cache.db"
        self.error_count = 0
        self.processed_count = 0

        if self.use_cache:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_llm(self):
        self.client: AsyncVLLMClient = AsyncVLLMClient(self.config)
        self.schema_dict = CtxsRelevanceParser.model_json_schema()

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
    
    def _prepare_input(self, query: str, ans_list: List[str], contexts: List[CtxExample]) -> str:
        joined_answer = ", ".join(ans_list)
        formatted_ctx = "\n".join([f"[{i}]\nTitle: {ctx.title}\n\n{ctx.text}\n" for i, ctx in enumerate(contexts)])
        
        user_content = self.prompt["user"].format(
            query=query,
            ref_answer=joined_answer,
            formatted_contexts=formatted_ctx,
            last_index=len(contexts)-1
        )
        return user_content
    
    async def judge(self, query: str, answer: List[str], contexts: List[CtxExample]) -> CtxsRelevance:
        pass

    async def batch_judge(
        self,
        queries: List[str],
        answers: List[List[str]],
        contexts_list: List[List[CtxExample]],
    ) -> List[CtxsRelevance]:
        """
        Asynchronously judge the relevance
        """
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
            pydantic_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "relevance_schema",
                    "schema": self.schema_dict,
                    "strict": False
                }
            }
            try:
                outputs = await self.client._generate_batch(
                    inputs=prompts_to_run,
                    system_prompt=self.prompt["system"],
                    user_template=None,
                    disable_template=True,
                    response_format=pydantic_format,
                    # show_progress=False,
                    max_tokens=self.config.sampling_params.max_tokens
                )
            except Exception as e:
                logger.error(f"vLLM batch inference failed: {str(e)}")
                return []
            
            cache_uptdates = []
            for idx, prompt, output in zip(indices_to_run, prompts_to_run, outputs):
                indices = re.findall(r"\[(\d+)\]", prompt)
                max_idx = max((int(i) for i in indices), default=-1)
                try:
                    parsed_obj = CtxsRelevanceParser.model_validate_json(output)
                    final_obj = parsed_obj.to_dataclass()
                except Exception as e:
                    self.error_count += 1
                    final_obj = CtxsRelevance()
                self.processed_count += max_idx + 1

                results[idx] = final_obj

                if self.use_cache:
                    cache_uptdates.append((cache_key, json.dumps(asdict(results[idx]))))

            # Insert new cache entries in batch
            if self.use_cache and cache_uptdates:
                with sqlite3.connect(self.cache_path) as conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO cache (key, data) VALUES (?, ?)",
                        cache_uptdates
                    )
        return results