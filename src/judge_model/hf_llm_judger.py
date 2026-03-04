from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import sqlite3
import hashlib
from typing import List
from pathlib import Path
from omegaconf import DictConfig
import outlines

from .template import apply_template
from .llm_judger import LLMJudger, JudgeOutput, CtxsRelevance
from .judger_prompt import HUGGINGFACE

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

        if self.use_cache:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_llm(self):
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

    def sanitize_output(self, output: JudgeOutput, max_idx: int) -> JudgeOutput:
        """
        LLM이 생성한 인덱스 중 범위를 벗어난(max_idx 초과) 쓰레기 값을 제거
        """
        def filter_idxs(idxs):
            return [i for i in idxs if 0 <= i <= max_idx]

        output.ctx_relevance.positive = filter_idxs(output.ctx_relevance.positive)
        output.ctx_relevance.negative = filter_idxs(output.ctx_relevance.negative)
        output.ctx_relevance.irrelevant = filter_idxs(output.ctx_relevance.irrelevant)
        
        return output
    
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