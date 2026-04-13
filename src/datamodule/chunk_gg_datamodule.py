from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset   # Prevent OOM
from typing import List, Optional
from tqdm import tqdm
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from dataclasses import asdict
import logging
import math
import torch
import torch.nn.functional as F
import h5py

from utils import load_relevance_dataset, format_reference_answer, RelevanceQAExample

logger = logging.getLogger(__name__)

TOPK_RANGE = 10

class GGChunkDataset(Dataset):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = RelevanceQAExample.from_dict(self.data[idx])
        question = item.question
        # reference answer will be used for loss calculation
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
        flat_target_input_ids = []
        for ctx in item.ctxs:
            ctx_text = f"Title: {ctx.title}\n\n{ctx.text}"
            tokenized_ctx = self.llm_tokenizer(
                ctx_text,
                truncation=True,
                max_length=self.cfg.data.max_seq_length - q_len,
                return_tensors="pt",
                add_special_tokens=False
            )
            
            doc_input_ids.append(torch.cat([tokenized_ctx["input_ids"], torch.tensor([[self.llm_tokenizer.eos_token_id]])], dim=1).squeeze(0))
            source_input_ids.append(torch.cat([tokenized_ctx["input_ids"], question_inputs["input_ids"]], dim=1).squeeze(0))
            flat_target_input_ids.append(tokenized_ctx["input_ids"].squeeze(0))
            doclen_list.append(tokenized_ctx["input_ids"].shape[1])

        # Concate D1 + ... + Dk + Q + A as target input (Chunked version)
        C = math.ceil(len(doc_input_ids) / TOPK_RANGE)
        target_input_ids = []
        for i in range(C):
            chunk = flat_target_input_ids[i*TOPK_RANGE : (i+1)*TOPK_RANGE]
            chunk_input_ids = torch.cat(chunk, dim=0)
            chunk_input_ids = torch.cat([chunk_input_ids, question_inputs["input_ids"][0], answer_inputs["input_ids"][0]], dim=0)
            target_input_ids.append(chunk_input_ids)

        scores_oracle = None
        if self.use_precompute_table:
            scores_oracle = self.oracle_cache[str(item.idx)]

        return {
            "idx": item.idx,
            "doc_input_ids": doc_input_ids,                  # (c*k, max_seq_length)
            "source_input_ids": source_input_ids,             # (c*k, max_seq_length)
            "target_input_ids": target_input_ids,                          # (c, k * max_seq_length + max_ans_length,)
            "doclen_list": doclen_list,                       # (c*k,)
            "a_len": a_len,                                   # (1,)   
            "question_ids": roberta_question_inputs["input_ids"].squeeze(0),                # (q_len,)
            "question_attention_mask": roberta_question_inputs["attention_mask"].squeeze(0),  # (q_len,)
            "scores_oracle": scores_oracle,  # (c*k,) or None
        }


class GGChunkDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.batch_size = self.data_cfg.batch_size
        self.num_workers = self.data_cfg.num_workers
        self.pad_token_id = 128004

    def setup(self, stage: Optional[str] = None):
        full_data = load_relevance_dataset(self.data_cfg.data_path)
        if self.cfg.debug_mode:
            full_data = full_data[:5000]
            
        self.oracle_cache = {}
        if self.data_cfg.use_precompute_table:
            score_mode = getattr(self.cfg.train, "oracle_mode", "marginal")
            logger.info("Loading pre-compute table entirely into RAM...")
            with h5py.File(self.data_cfg.precompute_table_path, "r") as f:
                for key in f.keys():
                    self.oracle_cache[key] = torch.tensor(f[key][score_mode][:], dtype=torch.bfloat16)
            
        dataset = [asdict(item) for item in tqdm(full_data, desc="Converts to dict")]
        hf_dataset = HFDataset.from_list(dataset)
        split_dataset = hf_dataset.train_test_split(test_size=self.data_cfg.test_size, seed=self.data_cfg.seed)
        if stage == 'fit' or stage is None:
            self.train_dataset = GGChunkDataset(split_dataset["train"], self.oracle_cache, self.cfg)
            self.val_dataset = GGChunkDataset(split_dataset["test"], self.oracle_cache, self.cfg)
        elif stage == 'validate':
            self.val_dataset = GGChunkDataset(split_dataset["test"], self.oracle_cache, self.cfg)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
    def collate_fn(self, batch):
        """
        Collate function to combine multiple docs into a single batch.
        """
        B = len(batch)
        TOTAL_K = len(batch[0]["doc_input_ids"])
        C = math.ceil(TOTAL_K / TOPK_RANGE)
        
        max_doc_len = max(len(d) for item in batch for d in item["doc_input_ids"])
        max_source_len = max(len(s) for item in batch for s in item["source_input_ids"])
        max_tgt_len = max(len(t) for item in batch for t in item["target_input_ids"])
        max_q_len = max(len(item["question_ids"]) for item in batch)

        flat_doc_ids = torch.full((B * TOTAL_K, max_doc_len), self.pad_token_id, dtype=torch.long)
        flat_doc_mask = torch.zeros((B * TOTAL_K, max_doc_len), dtype=torch.long)
        flat_source_ids = torch.full((B * TOTAL_K, max_source_len), self.pad_token_id, dtype=torch.long)
        flat_source_mask = torch.zeros((B * TOTAL_K, max_source_len), dtype=torch.long)

        padded_target_ids = torch.full((B, C, max_tgt_len), self.pad_token_id, dtype=torch.long)
        padded_target_mask = torch.zeros((B, C, max_tgt_len), dtype=torch.long)
        target_labels = torch.full((B, C, max_tgt_len), -100, dtype=torch.long)

        padded_question_ids = torch.full((B, max_q_len), self.pad_token_id, dtype=torch.long)
        padded_question_mask = torch.zeros((B, max_q_len), dtype=torch.long)

        for i, item in enumerate(batch):
            q_ids = item["question_ids"]
            q_len = len(q_ids)
            padded_question_ids[i, :q_len] = q_ids if isinstance(q_ids, torch.Tensor) else torch.tensor(q_ids)
            padded_question_mask[i, :q_len] = 1
            a_len = int(item["a_len"].item() if isinstance(item["a_len"], torch.Tensor) else item["a_len"])
            
            # ----- Target -----
            for c in range(C):
                t_ids = item["target_input_ids"][c]
                t_len = len(t_ids)
                padded_target_ids[i, c, :t_len] = t_ids[:t_len]
                padded_target_mask[i, c, :t_len] = 1
                
                valid_a_len = min(a_len, t_len)
                target_labels[i, c, t_len - valid_a_len : t_len] = padded_target_ids[i, c, t_len - valid_a_len : t_len]
                
            # ----- Doc & Src -----
            for j in range(TOTAL_K):
                idx = i * TOTAL_K + j

                # Doc
                d_ids = item["doc_input_ids"][j]
                d_len = len(d_ids)
                flat_doc_ids[idx, :d_len] = d_ids[:d_len]
                flat_doc_mask[idx, :d_len] = 1
                
                # Source
                s_ids = item["source_input_ids"][j]
                s_len = len(s_ids)
                flat_source_ids[idx, :s_len] = s_ids[:s_len]
                flat_source_mask[idx, :s_len] = 1

        use_scores_oracle = batch[0]["scores_oracle"] is not None
        
        return {
            "idx": torch.tensor([item["idx"] for item in batch]),    # (B,)
            "doc_input_ids": flat_doc_ids,                   # (B * TOTAL_K, max_doc_len)
            "doc_attention_mask": flat_doc_mask,             # (B * TOTAL_K, max_doc_len)
            "source_input_ids": flat_source_ids,             # (B * TOTAL_K, max_source_len)
            "source_attention_mask": flat_source_mask,       # (B * TOTAL_K, max_source_len)
            "target_input_ids": padded_target_ids,           # (B, C, max_tgt_len)
            "target_attention_mask": padded_target_mask,     # (B, C, max_tgt_len)
            "doclen_list": torch.tensor([item["doclen_list"] for item in batch]),                  # (B, TOTAL_K)
            "a_len": torch.tensor([item["a_len"] for item in batch]),                              # (B,)
            "question_ids": padded_question_ids,                 # (B, q_len)
            "question_attention_mask": padded_question_mask,     # (B, q_len)
            "scores_oracle": torch.stack([item["scores_oracle"] for item in batch]) if use_scores_oracle else None,  # (B, TOTAL_K) or None
        }

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
    data_module = GGChunkDataModule(cfg)
    data_module.setup("fit")
    for batch in data_module.train_dataloader():
        import pdb; pdb.set_trace()
        print(batch)
        break