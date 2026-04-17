from omegaconf import DictConfig
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import logging
from transformers import BatchEncoding, AutoTokenizer

from .load import load_model
from .prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from utils import CtxExample, template
from ..ca_former import CAFormerGGClassifier

logger = logging.getLogger(__name__)


class DISCA:
    """
    Decoding Intervention through ~~~ framework.
    """
    def __init__(self, config: DictConfig, model_name: str, caformer_clf: CAFormerGGClassifier) -> None:
        self.config: DictConfig = config
        self.model, self.tokenizer = load_model(model_name)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model_name: str = model_name
        # self.hidden_extraction_layers: List[int] = config.model.hidden_extraction_layers
        self.caformer_clf = caformer_clf.to(self.model.device)
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
            "use_cache": True,
        }
        ### Pad token
        self.pad_token = None
        if self.tokenizer.pad_token_id is not None:
            self.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
            

    @property
    def device(self):
        return self.model.device
    
    def encode(self, text: Union[str, List[str]], return_tokens_only: bool = True, max_len: int = 256) -> Union[torch.Tensor, BatchEncoding]:
        if isinstance(text, str):
            text = [text]
        
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
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
    
    def roberta_encode(self, text: Union[str, List[str]], return_tokens_only: bool = True, max_len: int = 256) -> Union[torch.Tensor, BatchEncoding]:
        if isinstance(text, str):
            text = [text]
        
        encoded = self.roberta_tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )

        if return_tokens_only:
            return encoded.input_ids.to(self.device)
        return encoded

    def build_inputs(
        self,
        queries: List[str],
        contexts_list: List[List[CtxExample]],
        use_prompt: bool = True
    ) -> Dict[str, Any]:
        """
        Build batch inputs for the model.
        Returns:
            - input_ids: Tensor of shape (batch_size, seq_len)
            - attention_mask: Tensor of shape (batch_size, seq_len)
            - source_ids: Tensor of shape (batch_size * num_contexts, seq_len)
            - source_attention_mask: Tensor of shape (batch_size * num_contexts, seq_len)
            - doclen_list: List of lists, where each inner list contains the lengths of contexts
        """
        # Construct input_ids with prompt template
        full_texts = []
        source_texts = []
        doclen_list = []
        for i, (q_text, ctx_list) in enumerate(zip(queries, contexts_list)):
            # Append PAD token to the end of each context
            ctx_text = "\n\n".join([f"Title: {ctx.title}\n\n{ctx.text}{self.pad_token}" for ctx in ctx_list])
            if use_prompt:
                query_text = self.generate_prompt.format(question=q_text)
                full_text = f"{self.sys_prompt_text}\n\n{ctx_text}\n\n{query_text}{self.postfix_text}"
            else:
                query_text = q_text
                full_text = f"{ctx_text}\n\n{query_text}"
            # Generate source text
            for ctx in ctx_list:
                source_text = f"Title: {ctx.title}\n\n{ctx.text}{self.pad_token}\n\n{query_text}"
                source_texts.append(source_text)
            full_texts.append(full_text)
        
        encoded = self.encode(full_texts, return_tokens_only=False, max_len=self.config.data.max_length).to(self.device)
        source_encoded = self.encode(source_texts,  return_tokens_only=False, max_len=self.config.data.max_seq_length).to(self.device)
        question_encoded = self.roberta_encode(queries, return_tokens_only=False, max_len=self.config.data.max_query_length).to(self.device)

        # Find the positions of PAD to get the length of each context
        if use_prompt:
            sys_prompt_dummy = f"{self.sys_prompt_text}\n\n"
            sys_prompt_len = len(self.tokenizer(sys_prompt_dummy, add_special_tokens=False).input_ids)
        else:
            sys_prompt_len = 0
        
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        pad_token_id = self.tokenizer.pad_token_id
        for b_idx in range(input_ids.size(0)):
            valid_ids = input_ids[b_idx][attention_mask[b_idx] == 1]
            marker_positions = (valid_ids == pad_token_id).nonzero(as_tuple=True)[0].tolist()
            b_doclens = []
            start_idx = sys_prompt_len

            for pos in marker_positions:
                # Include the PAD token
                doclen = pos - start_idx + 1
                b_doclens.append(doclen)
                start_idx = pos + 1
            
            # # Handle the last context
            # if start_idx < len(valid_ids):
            #     b_doclens.append(len(valid_ids) - start_idx)
            doclen_list.append(b_doclens)

        encoded_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "source_ids": source_encoded.input_ids,
            "source_attention_mask": source_encoded.attention_mask,
            "question_ids": question_encoded.input_ids,
            "question_attention_mask": question_encoded.attention_mask,
            "doclen_list": doclen_list,
        }

        return encoded_dict
    
    @torch.inference_mode()
    def intervene(self, inputs: Dict[str, Any], use_prompt=True) -> Dict[str, Any]:
        """
        Intervene on the model's hidden states using CA-Former
        """
        # Generate LLM hidden states for CA-Former input
        doc_repr = self.model.get_input_embeddings()(inputs["source_ids"])  # (B*K, L_doc, D_llm)
        llm_outputs = self.model(
            input_ids=inputs["source_ids"],
            attention_mask=inputs["source_attention_mask"],
            output_hidden_states=True,
        )
        llm_hidden_states = llm_outputs.hidden_states[-12:]
        del llm_outputs     # Free up memory
        llm_repr = torch.stack(llm_hidden_states).permute(1, 0, 2, 3)   # (B*K, L, S, D_llm)
        scores_hat, query_hidden_states = self.caformer_clf(llm_repr, inputs["source_attention_mask"],
                                                            inputs["question_ids"], inputs["question_attention_mask"])  # (B*K, K, D_probe), (B*K, K, D_llm)
        # Make interleaving inputs for generation
        if use_prompt:
            sys_prompt_dummy = f"{self.sys_prompt_text}\n\n"
            sys_input_ids = self.tokenizer(sys_prompt_dummy, add_special_tokens=False, return_tensors="pt").input_ids
            sys_prompt_len = len(sys_input_ids)
            sys_prompt_repr = self.model.get_input_embeddings()(sys_input_ids[0].to(self.device))  # (L_sys, D_llm)
        else:
            sys_prompt_len = 0
            sys_prompt_repr = None

        B = inputs["input_ids"].size(0)
        K = llm_repr.size(0) // B
        doclen_list = inputs["doclen_list"]
        inputs_embeds = []

        for i in range(B):
            b_inputs_embeds = []
            # Append system prompt representation if exists
            pad_offset = (inputs["attention_mask"][i] == 0).sum().item()
            start_pos = pad_offset + sys_prompt_len
            if sys_prompt_len > 0:
                b_inputs_embeds.append(sys_prompt_repr)
            # Interleave document representations and query representation
            for j in range(K):
                flat_idx = i*K + j
                doclen = doclen_list[i][j]
                source_pad_offset = (inputs["source_attention_mask"][flat_idx] == 0).sum().item()
                each_doc_repr = doc_repr[flat_idx, source_pad_offset:source_pad_offset + doclen]  # (doclen, D_llm)
                each_repr = torch.cat([each_doc_repr, query_hidden_states[flat_idx]], dim=0)
                b_inputs_embeds.append(each_repr)
                start_pos += doclen
            # Append the remaining P + A part representation
            full_embeds = self.model.get_input_embeddings()(inputs["input_ids"][i])
            if start_pos < full_embeds.size(0):
                remaining_repr = full_embeds[start_pos:]
                b_inputs_embeds.append(remaining_repr)
            b_inputs_embeds = torch.cat(b_inputs_embeds, dim=0)
            inputs_embeds.append(b_inputs_embeds)

        # Pad inputs_embeds to the same length
        max_len = max([emb.size(0) for emb in inputs_embeds])
        padded_inputs_embeds = []
        padded_attention_masks = []
        for inputs_embed in inputs_embeds:
            seq_len = inputs_embed.size(0)
            pad_len = max_len - seq_len
            # Do left padding
            if pad_len > 0:
                D_llm = inputs_embed.size(1)
                pad_emb = torch.zeros((pad_len, D_llm), device=inputs_embed.device, dtype=inputs_embed.dtype)
                padded_inputs_embed = torch.cat([pad_emb, inputs_embed], dim=0)
                padded_attention_mask = torch.ones((max_len,), device=inputs_embed.device, dtype=torch.long)
                padded_attention_mask[:pad_len] = 0
            else:
                padded_inputs_embed = inputs_embed
                padded_attention_mask = torch.ones((max_len,), device=inputs_embed.device, dtype=torch.long)
            padded_inputs_embeds.append(padded_inputs_embed)
            padded_attention_masks.append(padded_attention_mask)

        padded_inputs_embeds = torch.stack(padded_inputs_embeds, dim=0)   # (B, max_len, D_llm)
        padded_attention_masks = torch.stack(padded_attention_masks, dim=0)   # (B, max_len)

        return {
            "inputs_embeds": padded_inputs_embeds,
            "attention_mask": padded_attention_masks,
            "scores_hat": scores_hat,
        }
    
    @torch.inference_mode()
    def generate(
        self,
        queries: Union[str, List[str]],
        contexts_list: List[List[CtxExample]],
        do_intervene: bool = True
    ) -> List[str]:
        """
        Generate final answers using the generate prompt template.
        """
        if isinstance(queries, str):
            queries = [queries]

        inputs = self.build_inputs(queries, contexts_list)
        if do_intervene:
            inputs = self.intervene(inputs)
        else:
            inputs = {
                "inputs_embeds": self.model.get_input_embeddings()(inputs["input_ids"]),
                "attention_mask": inputs["attention_mask"],
            }
        
        # Generate answers, custom generation loop
        # prefill = self.model(
        #     inputs_embeds=inputs["inputs_embeds"],
        #     attention_mask=inputs["attention_mask"],
        # )
        # Greedy decoding
        generated_ids = self.model.generate(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            **self.gen_kwargs
        )
        answers = self.decode(generated_ids)
        return answers