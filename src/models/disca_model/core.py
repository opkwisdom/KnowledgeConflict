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
        prefix, postfix = template(self.model_name, task, base_template=False)
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
            "max_new_tokens": 32,
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
    
    def encode(self, text: Union[str, List[str]], return_tokens_only: bool = True, max_len: int = 256, add_special_tokens: bool = False) -> Union[torch.Tensor, BatchEncoding]:
        if isinstance(text, str):
            text = [text]
        
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
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
            formatted_query_text = f"\n\nQuestion: {q_text}"
            batch_query_texts.append(query_text)
            # Generate source text
            for ctx in ctx_list:
                doc_text = f"Title: {ctx.title}\n\n{ctx.text}"
                source_text = f"{doc_text}{formatted_query_text}"
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
    
    def rerank(self, inputs, scores_hat, topk):
        """
        Args:
            scores_hat: Tensor of shape (B*K, 1)
        Returns:
            reranked_doc_ids: Tensor of shape (B, topk, seq_len)
            reranked_doclen_tensor: Tensor of shape (B, topk)
            reranked_scores_hat: Tensor of shape (B, topk)
        """
        B = inputs["question_ids"].shape[0]
        K = inputs["source_ids"].shape[0] // B

        scores_hat = scores_hat.reshape(B, K)

        reranked_scores_hat, reranked_indices = torch.topk(scores_hat, k=topk, dim=1)   # (B, topk)
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
        }

        return reranked_dict
    
    def make_inputs(self, inputs, reranked_dict, use_prompt=True):
        B, TOPK, seq_len = reranked_dict["reranked_doc_ids"].shape
        batch_query_texts = inputs["batch_query_texts"]  # (B,)
        qlen_tensor = inputs["question_mask"].sum(dim=1)  # (B,)
        reranked_doclen_tensor = reranked_dict["reranked_doclen_tensor"]  # (B, topk)
        reranked_doc_ids = reranked_dict["reranked_doc_ids"]  # (B, topk, seq_len)

        newline_ids = self.encode("\n\n").view(-1).to(self.device)

        if use_prompt:
            sys_ids = self.sys_prompt_ids.to(self.device)         # (sys_len,)
            postfix_ids = self.postfix_ids.to(self.device)        # (postfix_len,)

            # Add generation prompt embeddings
            gen_prompt_encoded = self.encode(batch_query_texts, return_tokens_only=False)
            gen_prompt_lens = gen_prompt_encoded.attention_mask.sum(dim=1)  # (B,)
            gen_prompt_ids_full = gen_prompt_encoded.input_ids.to(self.device)  # (B, max_gen_len)
            gen_ids_list = [gen_prompt_ids_full[i, :gen_prompt_lens[i]] for i in range(B)]
        else:
            sys_ids = None
            postfix_ids = None
            gen_ids_list = None
        
        inputs_ids_list = []
        seq_lengths = []
        for i in range(B):
            b_ids = []
            # System prompt (prefix)
            if use_prompt:
                b_ids.append(sys_ids)

            for j in range(TOPK):
                each_doclen = reranked_doclen_tensor[i, j]
                each_doc_ids = reranked_doc_ids[i, j, -each_doclen:]    # Left-padding
                b_ids.append(each_doc_ids)
                b_ids.append(newline_ids)  # Add newline between contexts

            b_ids.append(newline_ids)
            # Generation prompt
            if use_prompt:
                b_ids.append(gen_ids_list[i])
                b_ids.append(postfix_ids)
            else:
                sample_question_ids = inputs["question_ids"][i, :qlen_tensor[i]]  # (qlen, D_llm)
                b_ids.append(sample_question_ids)
            
            b_ids = torch.cat(b_ids, dim=0)
            inputs_ids_list.append(b_ids)
            seq_lengths.append(b_ids.shape[0])
        
        inputs_ids = pad_sequence(inputs_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side='left')   # (B, max_len)
        attention_masks = [
            torch.ones(s, dtype=torch.long, device=inputs_ids.device)
            for s in seq_lengths
        ]
        attention_mask = pad_sequence(attention_masks, batch_first=True, padding_side='left')   # (B, max_len)
        return inputs_ids, attention_mask

    @torch.inference_mode()
    def intervene(self, inputs: Dict[str, Any], use_prompt=True) -> Dict[str, Any]:
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
        scores_hat, _ = self.caformer_clf(llm_repr, inputs["source_mask"],
                                        inputs["roberta_question_ids"], inputs["roberta_question_mask"])  # (B*K, 1), (B*K, M, D_llm)
        del llm_outputs, llm_repr     # Free up memory

        # Reranking
        reranked_dict = self.rerank(inputs, scores_hat, self.topk)

        # Make interleaving inputs for generation
        inputs_ids, attention_mask = self.make_inputs(inputs, reranked_dict, use_prompt)

        intervened_inputs = {
            "input_ids": inputs_ids,
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

        inputs = self.build_inputs(queries, contexts_list, use_prompt=True)
        if do_intervene:
            inputs = self.intervene(inputs, use_prompt=True)
        else:
            inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        # Greedy decoding
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            **self.gen_kwargs
        )
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]

        answers = self.decode(generated_ids)
        return answers