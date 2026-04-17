from openai import OpenAI
from typing import List
import os
import re
import sqlite3
import json
import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from omegaconf import DictConfig

from utils import CtxExample
from .template import apply_template
from .llm_judger import LLMJudger, JudgeOutput, CtxsRelevanceParser, CtxsRelevance
from .judger_prompt import OPENAI

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.08},
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25}
}

logger = logging.getLogger(__name__)

class OpenAIJudger(LLMJudger):
    """
    OpenAI LLM Judger implementation
    """
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.prompt = self.set_prompt(config.prompt_name)
        self.total_cost = 0.0
        self.llm_model_name = config.llm_model_name
        self.use_cache = config.use_cache
        self.cache_path = Path(config.cache_dir) / "judge_cache.db"

        self.error_count = 0
        self.processed_count = 0
        if self.use_cache:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_llm(self):
        self.client = OpenAI(api_key=os.getenv("JUNHO_OPENAI_API_KEY"))

    def _init_db(self):
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT
                )
            """)

    def _calculate_cost(self, usage, model_name) -> float:
        price_info = PRICING.get(model_name)
        if not price_info:
            return 0.0

        input_cost = (usage.prompt_tokens / 1_000_000) * price_info["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * price_info["output"]
        return input_cost + output_cost
    
    def _get_cache_key(self, user_content: str) -> str:
        unique_str = f"{self.llm_model_name}|{self.prompt['system']}|{user_content}"
        return hashlib.sha256(unique_str.encode("utf-8")).hexdigest()

    def set_prompt(self, prompt: str):
        return OPENAI.get(prompt, OPENAI["single_context_eval"])
    
    def _prepare_input(self, query: str, ans_list: List[str], contexts: List[CtxExample]) -> str:
        joined_answer = ", ".join(ans_list)
        formatted_ctx = "\n".join([f"[{i}]\nTitle: {ctx.title}\n\n{ctx.text}\n" for i, ctx in enumerate(contexts)])
        prompt_input = self.prompt["user"].format(
            query=query,
            ref_answer=joined_answer,
            formatted_contexts=formatted_ctx,
            last_index=len(contexts)-1
        )
        
        return prompt_input

    def judge(self, query: str, ans_list: List[str], contexts: List[CtxExample]) -> CtxsRelevance:
        prompt_input = self._prepare_input(query, ans_list, contexts)
        cache_key = self._get_cache_key(prompt_input)
        max_idx = max(int(i) for i in re.findall(r'\[(\d+)\]', prompt_input))
        self.processed_count += max_idx + 1
        
        # if self.use_cache:
        #     with sqlite3.connect(self.cache_path) as conn:
        #         cursor = conn.execute("SELECT data FROM cache WHERE key = ?", (cache_key,))
        #         row = cursor.fetchone()
        #         if row:
        #             cached_dict = json.loads(row[0])
        #             return CtxsRelevance(**cached_dict)
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": self.prompt["system"]},
                    {"role": "user", "content": prompt_input}
                ],
                response_format=CtxsRelevanceParser,
                temperature=0.0
            )
            usage = completion.usage
            cost = self._calculate_cost(usage, self.llm_model_name)
            self.total_cost += cost

            parsed_output = completion.choices[0].message.parsed
            parsed_data = parsed_output.to_dataclass()
            
            # Save to cache
            if self.use_cache:
                json_str = json.dumps(asdict(parsed_data))
                with sqlite3.connect(self.cache_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, data) VALUES (?, ?)",
                        (cache_key, json_str)
                    )

            return parsed_data
        
        except Exception as e:
            self.error_count += 1
            print(f"Error during OpenAI API call: {e}")
            raise e


    def get_total_cost(self) -> float:
        return self.total_cost