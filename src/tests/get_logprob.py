from models import AsyncVLLMClient
from omegaconf import DictConfig

llm = AsyncVLLMClient(DictConfig({}))
question = ["What is the capital of France?"]
output = llm.run(question, logprobs=True)[0]
content, all_logprobs = output
import pdb; pdb.set_trace()
x=1