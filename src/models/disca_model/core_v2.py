from omegaconf import DictConfig
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import logging
from transformers import BatchEncoding, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

from .load import load_model
from src.prompt import GENERATE_PROMPT
from utils import CtxExample, template
from ..ca_former import CAFormerGGClassifier

logger = logging.getLogger(__name__)


class DISCA:
    """
    Distillation-based Integrated Scoring via CA-Former
    """
    def __init__(self, config: DictConfig, model_name: str, caformer_clf: CAFormerGGClassifier) -> None:
        self.config: DictConfig = config
        self.model, self.tokenizer = load_model(model_name)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model_name: str = model_name
        # self.hidden_extraction_layers: List[int] = config.model.hidden_extraction_layers
        self.caformer_clf = caformer_clf.to(self.model.device)
        self.topk = config.data.topk_per_query
        self.__post_init__()

    def set_base_chat_template(self, task: str = "qa"):
        # Use base template for internal answer generation
        prefix, postfix = template(self.model_name, task, base_template=True)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix)[0], self.encode(postfix)[0]
        self.sys_prompt_text = prefix
        self.postfix_text = postfix

    def __post_init__(self):
        self.model.eval()
        self.set_base_chat_template()
        self.tokenizer.padding_side = "left"
        ### Set prompt templates
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
        ### Sequence length
        self.max_seq_length = self.config.data.max_seq_length
        self.max_query_length = self.config.data.max_query_length

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
            - doc_ids: Tensor of shape (B, K, seq_len)
            - doclen_tensor: Tensor of shape (B, K)
            - source_ids: Tensor of shape (B * K, seq_len + query_len)
            - source_mask: Tensor of shape (B * K, seq_len + query_len)
            - question_ids: Tensor of shape (B, query_len)
            - question_mask: Tensor of shape (B, query_len)
            - roberta_question_ids: Tensor of shape (B, query_len) for CA-Former classifier
            - roberta_question_mask: Tensor of shape (B, query_len) for CA-Former classifier
            - batch_query_texts: List of original query texts (B,)
        """
        # Construct input_ids with prompt template
        batch_query_texts = []    # (B,)
        batch_doc_texts = []      # (B, K)
        source_texts = []
        # doclen_list = []
        for i, (q_text, ctx_list) in enumerate(zip(queries, contexts_list)):
            query_text = self.generate_prompt.format(question=q_text) if use_prompt else q_text
            batch_query_texts.append(query_text)
            # Generate source text
            for ctx in ctx_list:
                doc_text = f"Title: {ctx.title}\n\n{ctx.text}"
                source_text = f"Title: {ctx.title}\n\n{ctx.text}{query_text}"
                batch_doc_texts.append(doc_text)
                source_texts.append(source_text)
        
        B = len(batch_query_texts)
        doc_encoded = self.encode(batch_doc_texts, return_tokens_only=False, max_len=self.max_seq_length).to(self.device)
        doc_ids = doc_encoded.input_ids.reshape(B, -1, doc_encoded.input_ids.size(-1))  # (B, K, seq_len)
        doclen_tensor = doc_encoded.attention_mask.sum(dim=1).reshape(B, -1)  # (B, K)
        source_encoded = self.encode(source_texts, return_tokens_only=False, max_len=self.max_seq_length + self.max_query_length).to(self.device)
        
        question_encoded = self.encode(queries, return_tokens_only=False, max_len=self.max_query_length).to(self.device)
        roberta_question_encoded = self.roberta_encode(queries, return_tokens_only=False, max_len=self.max_query_length).to(self.device)

        # Find the positions of PAD to get the length of each context
        # if use_prompt:
        #     sys_prompt_dummy = f"{self.sys_prompt_text}\n\n"
        #     sys_prompt_len = len(self.tokenizer(sys_prompt_dummy, add_special_tokens=False).input_ids)
        # else:
        #     sys_prompt_len = 0

        encoded_dict = {
            "doc_ids": doc_ids,                                                         # (B, K, seq_len)
            "doclen_tensor": doclen_tensor,                                             # (B, K)
            "source_ids": source_encoded.input_ids,                                     # (B*K, L_doc)
            "source_mask": source_encoded.attention_mask,                     # (B*K, L_doc)
            "question_ids": question_encoded.input_ids,                                 # (B, L_query)
            "question_mask": question_encoded.attention_mask,                 # (B, L_query)
            "roberta_question_ids": roberta_question_encoded.input_ids,                 # (B, L_query)
            "roberta_question_mask": roberta_question_encoded.attention_mask, # (B, L_query)
            "batch_query_texts": batch_query_texts,                                     # (B,)
        }

        return encoded_dict
    
    def rerank(self, inputs, scores_hat, query_hidden_states, topk):
        """
        Args:
            scores_hat: Tensor of shape (B*K, 1)
            query_hidden_states: Tensor of shape (B*K, M, D_llm)
        Returns:
            reranked_doc_ids: Tensor of shape (B, topk, seq_len)
            reranked_doclen_tensor: Tensor of shape (B, topk)
            reranked_scores_hat: Tensor of shape (B, topk)
            reranked_query_hidden_states: Tensor of shape (B, topk, M, D_llm)
        """
        B = inputs["question_ids"].shape[0]
        K = inputs["source_ids"].shape[0] // B

        _, M, D_llm = query_hidden_states.shape
        scores_hat = scores_hat.reshape(B, K)
        query_hidden_states = query_hidden_states.reshape(B, K, M, D_llm)

        reranked_scores_hat, reranked_indices = torch.topk(scores_hat, k=topk, dim=1)   # (B, topk)
        reranked_query_hidden_states = torch.gather(
            query_hidden_states, dim=1, index=reranked_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, D_llm)
        )      # (B, topk, M, D_llm)
        seq_len = inputs["doc_ids"].size(-1)
        reranked_doc_ids = torch.gather(
            inputs["doc_ids"], dim=1, index=reranked_indices.unsqueeze(-1).expand(-1, -1, seq_len)
        )      # (B, topk, seq_len)
        reranked_doclen_tensor = torch.gather(
            inputs["doclen_tensor"], dim=1, index=reranked_indices
        )  # (B, topk)

        reranked_dict = {
            "reranked_doc_ids": reranked_doc_ids,
            "reranked_doclen_tensor": reranked_doclen_tensor,
            "reranked_scores_hat": reranked_scores_hat,
            "reranked_query_hidden_states": reranked_query_hidden_states,
        }

        return reranked_dict
    
    def make_interleave_inputs(self, inputs, reranked_dict, use_prompt=False, use_caformer_prompt: bool = True):
        B, TOPK, seq_len = reranked_dict["reranked_doc_ids"].shape
        batch_query_texts = inputs["batch_query_texts"]  # (B,)
        question_embeds = self.model.get_input_embeddings()(inputs["question_ids"]) # (B, L_query, D_llm)
        qlen_tensor = inputs["question_mask"].sum(dim=1)  # (B,)
        reranked_doclen_tensor = reranked_dict["reranked_doclen_tensor"]  # (B, topk)
        reranked_query_hidden_states = reranked_dict["reranked_query_hidden_states"]  # (B, topk, M, D_llm)

        reranked_doc_embeds = self.model.get_input_embeddings()(
            reranked_dict["reranked_doc_ids"].reshape(-1, seq_len)
        ).reshape(B, TOPK, seq_len, -1)  # (B, topk, seq_len, D_llm)

        if use_prompt:
            # Add system prompt embeddings at the beginning
            sys_prompt_dummy = f"{self.sys_prompt_text}\n\n"
            sys_prompt_tokens = self.encode(sys_prompt_dummy).view(-1)
            sys_embeds = self.model.get_input_embeddings()(sys_prompt_tokens)
            # Add generation prompt embeddings at the end
            gen_prompt = [
                self.generate_prompt.format(question=query)
                for query in batch_query_texts
            ]
            gen_prompt_encoded = self.encode(gen_prompt, return_tokens_only=False)
            gen_prompt_lens = gen_prompt_encoded.attention_mask.sum(dim=1)  # (B,)
            gen_embeds_tensor = self.model.get_input_embeddings()(
                gen_prompt_encoded.input_ids.to(self.device)
            )
            gen_embeds = [gen_embeds_tensor[i, :gen_prompt_lens[i]] for i in range(B)]
        else:
            sys_embeds = None
            gen_embeds = None
        
        inputs_embeds = []
        seq_lengths = []

        for i in range(B):
            b_inputs_embeds = []
            # System prompt
            if use_prompt:
                b_inputs_embeds.append(sys_embeds)

            for j in range(TOPK):
                each_doclen = reranked_doclen_tensor[i, j]
                each_doc_embed = reranked_doc_embeds[i, j, :each_doclen]
                b_inputs_embeds.append(each_doc_embed)

                if use_caformer_prompt:
                    each_query_hidden_states = reranked_query_hidden_states[i, j]
                    b_inputs_embeds.append(each_query_hidden_states)
            # Generation prompt
            if use_prompt:
                sample_gen_embeds = gen_embeds[i]
                b_inputs_embeds.append(sample_gen_embeds)
            else:
                sample_question_embeds = question_embeds[i, :qlen_tensor[i]]  # (qlen, D_llm)
                b_inputs_embeds.append(sample_question_embeds)
            
            b_inputs_embeds = torch.cat(b_inputs_embeds, dim=0)
            inputs_embeds.append(b_inputs_embeds)
            seq_lengths.append(b_inputs_embeds.shape[0])
        
        inputs_embeds = pad_sequence(inputs_embeds, batch_first=True, padding_side='left')   # (B, max_len, D_llm)
        max_len = inputs_embeds.shape[1]
        seq_lengths_tensor = torch.tensor(seq_lengths, device=inputs_embeds.device)

        positions = torch.arange(max_len, device=inputs_embeds.device).unsqueeze(0) # (1, max_len)
        padding_lengths = max_len - seq_lengths_tensor.unsqueeze(1) # (B, 1)
        attention_mask = (positions >= padding_lengths).long()  # (B, max_len)
        return inputs_embeds, attention_mask

    @torch.inference_mode()
    def intervene(self, inputs: Dict[str, Any], use_prompt=False) -> Dict[str, Any]:
        """
        Intervene on the model's hidden states using CA-Former
        """
        # Generate LLM hidden states for CA-Former input
        B = inputs["question_ids"].shape[0]
        K = inputs["source_ids"].shape[0] // B

        llm_outputs = self.model.model(
            input_ids=inputs["source_ids"],
            attention_mask=inputs["source_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )
        llm_repr = torch.stack(llm_outputs.hidden_states[-12:]).permute(1, 0, 2, 3)   # (B*K, L, S, D_llm)
        scores_hat, query_hidden_states = self.caformer_clf(llm_repr, inputs["source_mask"],
                                                            inputs["roberta_question_ids"], inputs["roberta_question_mask"])  # (B*K, 1), (B*K, M, D_llm)
        del llm_outputs, llm_repr     # Free up memory

        # Reranking
        reranked_dict = self.rerank(inputs, scores_hat, query_hidden_states, self.topk)

        # Make interleaving inputs for generation
        inputs_embeds, attention_mask = self.make_interleave_inputs(inputs, reranked_dict, use_prompt)

        intervened_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        return intervened_inputs
    
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
        # Greedy decoding
        generated_ids = self.model.generate(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            **self.gen_kwargs
        )
        answers = self.decode(generated_ids)
        return answers