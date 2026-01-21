from openai import OpenAI
from typing import List
import os
import sqlite3
import json
import hashlib
import logging
from pathlib import Path
from omegaconf import DictConfig


logger = logging.getLogger(__name__)

from utils import CtxExample
from .llm_judger import LLMJudger, JudgeOutput, CtxsRelevance
from .judger_prompt import OPENAI

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.08},
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25}
}


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

        if self.use_cache:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_llm(self):
        self.client = OpenAI(api_key=os.getenv("NLPLAB_OPENAI_API_KEY"))

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
        return OPENAI.get(prompt, OPENAI["base"])

    def judge(self, query: str, answer: str, contexts: List[CtxExample]) -> JudgeOutput:
        # Prepare the input prompt
        if( len(contexts) > 1):
            formatted_ctx = "\n".join([f"[{i}]\nTitle: {ctx.title}\n\n{ctx.text}\n" for i, ctx in enumerate(contexts)])
        else:
            single_ctx = contexts[0]
            formatted_ctx = f"[0]\nTitle: {single_ctx.title}\n\n{single_ctx.text}"
        
        format_args = {
            "query": query,
            "internal_answer": answer,
            "formatted_contexts": formatted_ctx,
        }
        if "single" not in self.config.prompt_name:
            format_args["last_index"] = len(contexts) - 1

        user_content = self.prompt["user"].format(**format_args)
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
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": self.prompt["system"]},
                    {"role": "user", "content": user_content}
                ],
                response_format=JudgeOutput,
                temperature=0.0
            )

            usage = completion.usage
            cost = self._calculate_cost(usage, self.llm_model_name)
            self.total_cost += cost

            parsed_output = completion.choices[0].message.parsed
            
            # Save to cache
            if self.use_cache:
                json_str = parsed_output.model_dump_json()
                with sqlite3.connect(self.cache_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, data) VALUES (?, ?)",
                        (cache_key, json_str)
                    )

            return parsed_output
        
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            raise e
    
    def get_total_cost(self) -> float:
        return self.total_cost
    
    def export_sqlite_to_json(self):
        output_json_path = self.cache_path.with_suffix('.json')
        
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, data FROM cache") # 순서: 0번 key, 1번 data
            rows = cursor.fetchall()

            validated_list = []

            for row in rows:
                # ★ 여기가 핵심 수정 사항입니다 ★
                # row[0]은 key(해시값), row[1]이 data(JSON)입니다.
                key, data_str = row 

                try:
                    # 해시값(key)이 아니라 데이터(data_str)를 파싱해야 합니다.
                    raw_dict = json.loads(data_str)
                    
                    # Pydantic 검증 및 변환
                    pydantic_obj = JudgeOutput.model_validate(raw_dict)
                    clean_dict = pydantic_obj.model_dump(mode='json')
                    
                    validated_list.append(clean_dict)

                except Exception as e:
                    print(f"⚠️ 에러 발생 (Key: {key}): {e}")

            # 저장
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(validated_list, f, ensure_ascii=False, indent=4)
                
            print(f"성공! {len(validated_list)}개의 데이터를 복구했습니다.")
