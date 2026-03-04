from omegaconf import DictConfig
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import logging
from transformers import BatchEncoding

from .load import load_model
from .prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from utils import CtxExample, template
from ..ca_former.modeling_ca_former import SingleHiddenCAFormer, MultiHiddenCAFormer

logger = logging.getLogger(__name__)


class DISCA:
    """
    Decoding Intervention through Self-Generated Conflict Amplification framework.
    """
    def __init__(self, config: DictConfig, model_name: str, caformer: Optional[Union[SingleHiddenCAFormer, MultiHiddenCAFormer]] = None) -> None:
        self.config: DictConfig = config
        self.model, self.tokenizer = load_model(model_name)
        self.model_name: str = model_name
        self.hidden_extraction_layers: List[int] = config.model.hidden_extraction_layers
        self.caformer = caformer
        self.__post_init__()
    
    def set_base_chat_template(self, task: str = "qa"):
        # Use base template for internal answer generation
        prefix, postfix = template(self.model_name, task, base_template=True)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix)[0], self.encode(postfix)[0]
        self.sys_prompt_text = prefix
        self.postfix_text = postfix
    
    def __post_init__(self):
        self.set_base_chat_template()
        self.tokenizer.padding_side = "left"
        ### Set prompt templates
        self.self_task_prompt: str = ALL_PROMPTS[self.config.self_task_prompt_name]     # Used for SCA prompt construction
        self.base_prompt: str = GENERATE_PROMPT[self.config.base_prompt_name]           # Used for internal answer generation
        self.generate_prompt: str = GENERATE_PROMPT[self.config.generate_prompt_name]   # Used for final answer generation
        ### Set generation kwargs
        self.gen_kwargs = self.config.model.gen_kwargs \
            if self.config.model.gen_kwargs is not None else {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1,
            "top_k": None,
            "max_new_tokens": 512,
        }

    @property
    def device(self):
        return self.model.device

    def encode(self, text: Union[str, List[str]], return_tokens_only: bool = True) -> Union[torch.Tensor, BatchEncoding]:
        if isinstance(text, str):
            text = [text]
        
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )

        if return_tokens_only:
            return encoded.input_ids.to(self.device)
        return encoded
    
    def decode(self, input_ids: torch.Tensor, is_special_prompt: bool = False) -> List[str]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        decoded = self.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=(not is_special_prompt),
        )
        return decoded
    
    def apply_template(self, queries: List[str]) -> torch.Tensor:
        raise NotImplementedError("method `apply_template` is not implemented yet.")
    
    def __call__(
        self,
        queries: List[str],
        contexts_list: List[List[CtxExample]],   
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Self-Generated Conflict Amplification (SCA) pipeline
        Returns:
            caformer_input: Tensor of shape (B, L_select, S, D)
            caformer_mask: Tensor of shape (B, S)
        """
        a_int_list = self.generate_internal_answers(queries)
        
        flat_input_texts = []
        max_target_length = -1
        target_length_list = []
        self_task_prompt_text = self.self_task_prompt

        # TODO: SCA batch prompt construction
        for i, (q_text, a_int, ctx_list) in enumerate(zip(queries, a_int_list, contexts_list)):
            self_task_prompt = f"{q_text}\n\n{a_int}\n\n{self_task_prompt_text}{self.postfix_text}{a_int}"
            self_task_prompt_ids = self.encode(self_task_prompt)[0]
            
            # Keep track of variable target lengths across different samples
            target_length = len(self_task_prompt_ids)
            target_length_list.append(target_length)
            max_target_length = max(max_target_length, target_length)
            for ctx in ctx_list:
                full_text = f"{self.sys_prompt_text}\n\nTitle: {ctx.title}\n\n{ctx.text}\n\n{self_task_prompt}"
                flat_input_texts.append(full_text)
        
        sca_encoded = self.encode(flat_input_texts, return_tokens_only=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**sca_encoded, use_cache=False, output_hidden_states=True)

        # Which hidden states to return?
        selected_hiddens = [
            outputs.hidden_states[layer_idx]
            for layer_idx in self.hidden_extraction_layers
        ]
        selected_hiddens = torch.stack(selected_hiddens, dim=1)     # (B, L_select, S, D)
        caformer_input = selected_hiddens[:, :, -max_target_length:, :]
        caformer_mask = torch.zeros_like(caformer_input[:, 0, :, 0], dtype=torch.long).to(self.device)
        
        for i, target_length in enumerate(target_length_list):
            start_idx = i * self.config.data.topk_per_query
            end_idx = start_idx + self.config.data.topk_per_query
            caformer_mask[start_idx:end_idx, -target_length:] = 1
        
        return caformer_input, caformer_mask
    
    @torch.inference_mode()
    def generate_internal_answers(
        self,
        queries: Union[str, List[str]],
    ) -> List[str]:
        """
        Generate internal answers using the base prompt template.
        """
        if isinstance(queries, str):
            queries = [queries]
        
        # Construct input_ids with prompt template
        full_prompts = []
        for query in queries:
            formatted_q = self.base_prompt.format(question=query)
            full_prompt = f"{self.sys_prompt_text}\n\n{formatted_q}{self.postfix_text}"
            full_prompts.append(full_prompt)
        encoded = self.encode(full_prompts, return_tokens_only=False).to(self.device)

        output = self.model.generate(**encoded, **self.gen_kwargs)
        input_len = encoded.input_ids.shape[1]
        gen_ids = output[:, input_len:]
        generated_text = self.decode(gen_ids)

        return generated_text
    
    ### TODO: implement the following two methods for inference ###
    @torch.inference_mode()
    def _generate_intervened(
        self,
        queries: Union[str, List[str]],
        contexts_list: List[List[CtxExample]],
    ) -> List[str]:
        """
        Generate intervened answers using CAFormer
        """
        raise NotImplementedError()
    
    @torch.inference_mode()
    def _generate_vanilla(
        self,
        queries: Union[str, List[str]],
        contexts_list: List[List[CtxExample]],
    ) -> List[str]:
        """
        Generate vanilla answers without intervention
        """
        raise NotImplementedError()
    
    @torch.inference_mode()
    def generate(
        self,
        queries: Union[str, List[str]],
        contexts_list: List[List[CtxExample]],
    ) -> List[str]:
        """
        Generate final answers using the generate prompt template.
        """
        if self.caformer is not None:
            return self._generate_intervened(queries, contexts_list)
        return self._generate_vanilla(queries, contexts_list)