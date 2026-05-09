from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset   # Prevent OOM
from typing import List, Optional
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from dataclasses import asdict
from src.prompt import GENERATE_PROMPT
from src.utils import apply_template
import logging
import torch
import torch.nn.functional as F
import h5py

from utils import load_qa_dataset, format_reference_answer, RelevanceQAExample

logger = logging.getLogger(__name__)


class GGDataset(Dataset):
    def __init__(self, data: HFDataset, oracle_cache: dict, cfg: DictConfig):
        self.data = data
        self.cfg = cfg
        self.oracle_cache = oracle_cache
        self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token_id = 128004
        self.llm_tokenizer.padding_side = "right"
        self.roberta_tokenizer.padding_side = "right"

        # Pre-compute table related attributes
        self.use_precompute_table = cfg.data.use_precompute_table
        self.generation_prompt_template = GENERATE_PROMPT["base"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        We don't use any padding here since we will do dynamic padding in the collate_fn.
        """
        item = RelevanceQAExample.from_dict(self.data[idx])
        question = item.question
        generation_prompt = "\n\n" + self.generation_prompt_template.format(question=question)

        # reference answer will be used for loss calculation
        # use pseudo answer? richer signal for CA-Former?
        if self.cfg.data.use_pseudo_answer:
            ref_answer = item.pseudo_answer
        else:
            ref_answer = format_reference_answer(item.answers)

        answer_inputs = self.llm_tokenizer(
            ref_answer,
            truncation=True,
            max_length=self.cfg.data.max_ans_length,
            return_tensors="pt",
            add_special_tokens=False
        )
        formatted_question = f"\n\nQuestion: {question}"
        question_inputs = self.llm_tokenizer(
            formatted_question,
            truncation=True,
            max_length=self.cfg.data.max_seq_length,
            return_tensors="pt",
            add_special_tokens=False
        )
        generation_prompt_inputs = self.llm_tokenizer(
            generation_prompt,
            truncation=True,
            max_length=self.cfg.data.max_seq_length,
            return_tensors="pt",
            add_special_tokens=False
        )
        q_len = question_inputs["input_ids"].shape[1]
        a_len = answer_inputs["input_ids"].shape[1]

        # Roberta question inputs
        roberta_question_inputs = self.roberta_tokenizer(
            formatted_question,
            truncation=True,
            max_length=self.cfg.data.max_seq_length,
            return_tensors="pt",
            add_special_tokens=False
        )

        # Construct document inputs and source inputs
        doclen_list = []
        doc_input_ids = []
        source_input_ids = []
        # ctxs = item.ctxs[:self.cfg.data.first_topk]  # First retrieval
        ctxs = item.ctxs[10:10+self.cfg.data.first_topk]  # Skip the first 10 passages, which contain golden passages
        for ctx in ctxs:
            ctx_text = f"Title: {ctx.title}\n\n{ctx.text}"
            tokenized_ctx = self.llm_tokenizer(
                ctx_text,
                truncation=True,
                max_length=self.cfg.data.max_seq_length - q_len,
                return_tensors="pt",
                add_special_tokens=False
            )
            
            doc_input = torch.cat([tokenized_ctx["input_ids"], torch.tensor([[self.llm_tokenizer.eos_token_id]])], dim=1)
            source_input = torch.cat([tokenized_ctx["input_ids"], question_inputs["input_ids"]], dim=1)
            
            doc_input_ids.append(doc_input.squeeze(0))
            source_input_ids.append(source_input.squeeze(0))
            doclen_list.append(tokenized_ctx["input_ids"].shape[1])
        
        scores_oracle = None
        if self.use_precompute_table:
            scores_oracle = self.oracle_cache[str(item.idx)]

        return {
            "idx": item.idx,
            "doc_input_ids": doc_input_ids,                   # (k, max_seq_length)
            "source_input_ids": source_input_ids,             # (k, max_seq_length)
            "doclen_list": doclen_list,                       # (k,)
            "a_len": a_len,                                   # (1,)   
            "q_len": q_len,
            "generation_prompt_ids": generation_prompt_inputs["input_ids"].squeeze(0),      # (gen_prompt_len,)
            "answer_ids": answer_inputs["input_ids"].squeeze(0),                            # (a_len,)
            "question_ids": question_inputs["input_ids"].squeeze(0),                        # (q_len,)
            "roberta_question_ids": roberta_question_inputs["input_ids"].squeeze(0),        # (roberta_q_len,)
            "roberta_question_mask": roberta_question_inputs["attention_mask"].squeeze(0),  # (roberta_q_len,)
            "scores_oracle": scores_oracle,  # (k,) or None
        }


class GGDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.batch_size = self.data_cfg.batch_size
        self.num_workers = self.data_cfg.num_workers
        self.pad_token_id = 128004
        self.roberta_pad_token_id = 1

    def setup(self, stage: Optional[str] = None):
        full_data = load_qa_dataset(self.data_cfg.data_path)
        if self.cfg.debug_mode:
            full_data = full_data[:5000]
            
        self.oracle_cache = {}
        if self.data_cfg.use_precompute_table:
            logger.info("Loading pre-compute table entirely into RAM...")
            with h5py.File(self.data_cfg.precompute_table_path, "r") as f:
                for key in f.keys():
                    # self.oracle_cache[key] = torch.tensor(f[key][score_mode][:], dtype=torch.bfloat16)
                    oracle_scores = f[key]['base_loss'][:] - f[key]['doc_loss'][:]
                    # Skip the first 10 passages, which contain golden passages, and take the next top-k passages for training
                    self.oracle_cache[key] = torch.tensor(oracle_scores[10:10+self.data_cfg.first_topk], dtype=torch.bfloat16)
            
        dataset = [asdict(item) for item in tqdm(full_data, desc="Converts to dict")]
        hf_dataset = HFDataset.from_list(dataset)
        split_dataset = hf_dataset.train_test_split(test_size=self.data_cfg.test_size, seed=self.data_cfg.seed)
        if stage == 'fit' or stage is None:
            self.train_dataset = GGDataset(split_dataset["train"], self.oracle_cache, self.cfg)
            self.val_dataset = GGDataset(split_dataset["test"], self.oracle_cache, self.cfg)
        elif stage == 'validate':
            self.val_dataset = GGDataset(split_dataset["test"], self.oracle_cache, self.cfg)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def collate_fn(self, batch):
        """
        Collate function to combine multiple docs into a single batch.
        """
        B = len(batch)
        K = len(batch[0]["doc_input_ids"])
        
        flat_doc_ids_list = []
        flat_source_ids_list = []
        roberta_q_ids_list = []

        doc_lens = []
        source_lens = []
        roberta_q_lens = []

        for item in batch:
            # ----- Question -----
            r_q_ids = torch.tensor(item["roberta_question_ids"], dtype=torch.long)
            roberta_q_ids_list.append(r_q_ids)
            roberta_q_lens.append(len(r_q_ids))

            # ----- Doc & Src -----
            for j in range(K):
                d_ids = torch.tensor(item["doc_input_ids"][j], dtype=torch.long)
                s_ids = torch.tensor(item["source_input_ids"][j], dtype=torch.long)
                
                flat_doc_ids_list.append(d_ids)
                flat_source_ids_list.append(s_ids)
                
                doc_lens.append(len(d_ids))
                source_lens.append(len(s_ids))

        # Utilize C++ backends
        flat_doc_ids = pad_sequence(flat_doc_ids_list, batch_first=True, padding_value=self.pad_token_id)
        flat_source_ids = pad_sequence(flat_source_ids_list, batch_first=True, padding_value=self.pad_token_id)
        padded_roberta_question_ids = pad_sequence(roberta_q_ids_list, batch_first=True, padding_value=self.roberta_pad_token_id)

        max_doc_len = flat_doc_ids.shape[1]
        max_source_len = flat_source_ids.shape[1]
        max_roberta_q_len = padded_roberta_question_ids.shape[1]

        doc_lens_tensor = torch.tensor(doc_lens)
        source_lens_tensor = torch.tensor(source_lens)
        roberta_q_lens_tensor = torch.tensor(roberta_q_lens)

        flat_doc_mask = (torch.arange(max_doc_len).unsqueeze(0) < doc_lens_tensor.unsqueeze(1)).long()
        flat_source_mask = (torch.arange(max_source_len).unsqueeze(0) < source_lens_tensor.unsqueeze(1)).long()
        padded_roberta_question_mask = (torch.arange(max_roberta_q_len).unsqueeze(0) < roberta_q_lens_tensor.unsqueeze(1)).long()

        use_scores_oracle = batch[0]["scores_oracle"] is not None

        return {
            "idx": [item['idx'] for item in batch],
            # Document input to be combined in gen loss
            "doc_input_ids": flat_doc_ids,
            "doc_attention_mask": flat_doc_mask,
            # Source input for CA-Former
            "source_input_ids": flat_source_ids,
            "source_attention_mask": flat_source_mask,
            # Meta-information which are needed after re-ranking and gen loss calculation
            "doclen_list": torch.tensor([item["doclen_list"] for item in batch]),
            "a_len": torch.tensor([item["a_len"] for item in batch]),
            # Generation prompt input
            "generation_prompt_ids": [item["generation_prompt_ids"] for item in batch],
            # Answer input
            "answer_ids": [item["answer_ids"] for item in batch],
            # Question input for gen loss after re-ranking (Not padded)
            "question_ids": [item["question_ids"] for item in batch],
            # Question input for CA-Former
            "roberta_question_ids": padded_roberta_question_ids,
            "roberta_question_mask": padded_roberta_question_mask,
            "scores_oracle": torch.stack([item["scores_oracle"] for item in batch]) if use_scores_oracle else None,
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            prefetch_factor=2,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            prefetch_factor=2,
        )

    # def collate_fn(self, batch):
    #     """
    #     Collate function to combine multiple docs into a single batch.
    #     """
    #     B = len(batch)
    #     K = len(batch[0]["doc_input_ids"])

    #     max_doc_len = max(len(d) for item in batch for d in item["doc_input_ids"])
    #     max_source_len = max(len(s) for item in batch for s in item["source_input_ids"])
    #     max_roberta_q_len = max(len(item["roberta_question_ids"]) for item in batch)

    #     flat_doc_ids = torch.full((B * K, max_doc_len), self.pad_token_id, dtype=torch.long)
    #     flat_doc_mask = torch.zeros((B * K, max_doc_len), dtype=torch.long)
    #     flat_source_ids = torch.full((B * K, max_source_len), self.pad_token_id, dtype=torch.long)
    #     flat_source_mask = torch.zeros((B * K, max_source_len), dtype=torch.long)

    #     # question ids will be used in later, and we should combine them with re-ranked doc_ids
    #     # padded_question_ids = torch.full((B, max_q_len), self.roberta_pad_token_id, dtype=torch.long)
    #     padded_roberta_question_ids = torch.full((B, max_roberta_q_len), self.roberta_pad_token_id, dtype=torch.long)
    #     padded_roberta_question_mask = torch.zeros((B, max_roberta_q_len), dtype=torch.long)

    #     # We do right padding
    #     for i, item in enumerate(batch):
    #         # ----- Question -----
    #         roberta_q_ids = item["roberta_question_ids"]
    #         roberta_q_len = len(roberta_q_ids)
    #         padded_roberta_question_ids[i, :roberta_q_len] = roberta_q_ids
    #         padded_roberta_question_mask[i, :roberta_q_len] = 1
    #         # ----- Doc & Src -----
    #         for j in range(K):
    #             idx = i*K + j

    #             # Doc
    #             d_ids = item["doc_input_ids"][j]
    #             d_len = len(d_ids)
    #             flat_doc_ids[idx, :d_len] = d_ids
    #             flat_doc_mask[idx, :d_len] = 1

    #             # Source
    #             s_ids = item["source_input_ids"][j]
    #             s_len = len(s_ids)
    #             flat_source_ids[idx, :s_len] = s_ids
    #             flat_source_mask[idx, :s_len] = 1

    #     use_scores_oracle = batch[0]["scores_oracle"] is not None

    #     return {
    #         "idx": torch.tensor([item['idx'] for item in batch]),    # (B,)
    #         # Document input to be combined in gen loss
    #         "doc_input_ids": flat_doc_ids,                   # (B * K, max_doc_len)
    #         "doc_attention_mask": flat_doc_mask,             # (B * K, max_doc_len)
    #         # Source input for CA-Former
    #         "source_input_ids": flat_source_ids,             # (B * K, max_source_len)
    #         "source_attention_mask": flat_source_mask,       # (B * K, max_source_len)
    #         # Meta-information which are needed after re-ranking and gen loss calculation
    #         "doclen_list": torch.tensor([item["doclen_list"] for item in batch]),                  # (B, K)
    #         "a_len": torch.tensor([item["a_len"] for item in batch]),                              # (B,)
    #         # Answer input
    #         "answer_ids": [item["answer_ids"] for item in batch],                                 # (B, a_len)
    #         # Question input for gen loss after re-ranking (Not padded)
    #         "question_ids": [item["question_ids"] for item in batch],                           # (B, q_len)
    #         # Question input for CA-Former
    #         "roberta_question_ids": padded_roberta_question_ids,                 # (B, roberta_q_len)
    #         "roberta_question_attention_mask": padded_roberta_question_mask,     # (B, roberta_q_len)
    #         "scores_oracle": torch.stack([item["scores_oracle"] for item in batch]) if use_scores_oracle else None,  # (B, K) or None
    #     }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/src/train/run_gg_train.yaml")
    
    # dataset = load_relevance_dataset(cfg.data.data_path)
    # hf_dataset = HFDataset.from_list([asdict(item) for item in tqdm(dataset, desc="Converts to dict")])
    # data = GGDataset(hf_dataset, cfg)
    # output = data[0]
    # import pdb; pdb.set_trace()
    # x=1
    data_module = GGDataModule(cfg)
    data_module.setup("fit")
    for batch in data_module.train_dataloader():
        import pdb; pdb.set_trace()
        print(batch)
        break