import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from omegaconf import DictConfig

### How to use vLLM API in terminal:
# echo "gpt-oss-120b() { ssh -p 55391 nlplab15@10.201.135.160 \"gpt-oss-120b '\$*' \"; }" >> ~/.zshrc && source ~/.zshrc
# If you cannot use the above command directly,
# 1. ssh-keygen
# 2. ssh-copy-id -p 55391 nlplab15@10.201.135.160


# First, set up SSH port forwarding to access the vLLM API server locally
# ssh -N -f -p 55391 -L 18000:localhost:8000 nlplab15@10.201.135.160

class AsyncVLLMClient:
    def __init__(self, config: DictConfig):
        """
        vLLM API 서버와 통신하는 비동기 클라이언트.
        """
        self.api_url = "http://localhost:18000/v1/chat/completions"
        concurrency_limit = getattr(config, "concurrency_limit", 128)
        self.model_name = getattr(config, "model_name", "openai/gpt-oss-120b")
        self.sem = asyncio.Semaphore(concurrency_limit)
        self.stop_tokens = getattr(config, "stop", ["<|return|>", "<|call|>", "<|end|>", "<|message|>"])
        
        self.default_system = (
            "You are an expert AI research scientist."
            "Your goal is to provide accurate, concise, and helpful answers to user inquiries while adhering to strict constraints."
        )
        self.default_user_template = "Question: {query}\nAnswer: "

    async def _fetch(self, session, input_data, system_prompt, user_template, **kwargs):
        if isinstance(input_data, dict) and user_template:
            user_content = user_template.format(**input_data)
        elif kwargs.get("disable_template", False):
            user_content = str(input_data)
        else:
            # If no template is provided
            user_content = self.default_user_template.format(query=str(input_data))

        sys_prompt = system_prompt if system_prompt else self.default_system

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": kwargs.get("temperature", 0.0), # 기본값 0.0 (단답형)
            "max_tokens": kwargs.get("max_tokens", 512),
            "stop": self.stop_tokens,
            "logprobs": kwargs.get("logprobs", False),
            "top_logprobs": kwargs.get("top_logprobs", 5),
        }

        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        async with self.sem:
            try:
                async with session.post(self.api_url, json=payload, timeout=kwargs.get("timeout", 300)) as response:
                    if response.status == 200:
                        res_data = await response.json()
                        content = res_data['choices'][0]['message']['content']
                        logprobs_info = res_data['choices'][0].get('logprobs', {}).get('content', [])
                        if kwargs.get("logprobs", False):
                            return content, logprobs_info
                        return content
                    else:
                        error_msg = await response.text()
                        return f"[Error HTTP {response.status}] {error_msg}"
            except Exception as e:
                return f"[Exception] {str(e)}"

    async def _generate_batch(self, inputs, system_prompt, user_template, **kwargs):
        show_progress = kwargs.get("show_progress", True)
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch(session, item, system_prompt, user_template, **kwargs) for item in inputs]
            if show_progress:
                return await tqdm.gather(*tasks, desc="vLLM Processing")
            else:
                return await asyncio.gather(*tasks)

    def run(self, inputs, system_prompt=None, user_template=None, **kwargs):
        return asyncio.run(self._generate_batch(inputs, system_prompt, user_template, **kwargs))