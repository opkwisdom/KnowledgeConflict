from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset   # Prevent OOM
from typing import List, Optional
from tqdm import tqdm
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from dataclasses import asdict
import logging
import torch
import torch.nn.functional as F

from utils import load_relevance_dataset, format_reference_answer, RelevanceQAExample

logger = logging.getLogger(__name__)


class GGDataset(Dataset):
    def __init__(self, data: HFDataset, cfg: DictConfig):
        self.data = data
        self.cfg = cfg
        self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token_id = 128004
        self.llm_tokenizer.padding_side = "right"

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

        # Construct document inputs and source inputs
        doclen_list = []
        doc_input_ids = []
        source_input_ids = []
        target_input_ids = []
        for ctx in item.ctxs:
            ctx_text = f"Title: {ctx.title}\n\n{ctx.text}"
            tokenized_ctx = self.llm_tokenizer(
                ctx_text,
                truncation=True,
                max_length=self.cfg.data.max_seq_length - q_len,
                return_tensors="pt",
                add_special_tokens=False
            )

            # Append document length for later use
            doclen_list.append(tokenized_ctx.attention_mask.sum().item())

            ### ========== Document input for extracting document hidden states ========== ###
            # Append EOS token to the end of each document input to extract hidden states of document
            tokenized_doc = torch.cat([tokenized_ctx["input_ids"], torch.tensor([[self.llm_tokenizer.eos_token_id]])], dim=1)
            doc_input_ids.append(tokenized_doc.squeeze(0))

            ### ========== Source input for CAFormer ========== ###
            # Append D + Q as source input
            tokenized_source = torch.cat([tokenized_ctx["input_ids"], question_inputs["input_ids"]], dim=1)
            source_input_ids.append(tokenized_source.squeeze(0))

            ### ========== Target input for loss calculation ========== ###
            # Append D1 + ... + Dk as target input (without question + answer)
            target_input_ids.append(tokenized_ctx["input_ids"].squeeze(0))
        
        # Concate D1 + ... + Dk + Q + A as target input
        target_input_ids = torch.cat(target_input_ids, dim=0)
        target_input_ids = torch.cat([target_input_ids, question_inputs["input_ids"][0], answer_inputs["input_ids"][0]], dim=0)
        target_attention_mask = torch.ones_like(target_input_ids)

        # Pad source inputs and document inputs to the same length, right padding
        pad_token_id = self.llm_tokenizer.pad_token_id

        padded_doc_ids = []
        padded_doc_masks = []
        padded_source_ids = []
        padded_source_masks = []

        for i in range(len(item.ctxs)):
            # Pad document input
            d_ids = doc_input_ids[i]
            d_mask = torch.ones_like(d_ids)
            
            pad_len_doc = max(0, self.cfg.data.max_seq_length - len(d_ids))
            
            padded_doc_ids.append(F.pad(d_ids, (0, pad_len_doc), value=pad_token_id))
            padded_doc_masks.append(F.pad(d_mask, (0, pad_len_doc), value=0))

            # Pad source input
            s_ids = source_input_ids[i]
            s_mask = torch.ones_like(s_ids)
            
            pad_len_src = max(0, self.cfg.data.max_seq_length - len(s_ids))
            padded_source_ids.append(F.pad(s_ids, (0, pad_len_src), value=pad_token_id))
            padded_source_masks.append(F.pad(s_mask, (0, pad_len_src), value=0))
        # Pad target input
        pad_len_tgt = max(0, self.cfg.data.topk_per_query * self.cfg.data.max_seq_length + self.cfg.data.max_ans_length - len(target_input_ids))
        padded_target_ids = F.pad(target_input_ids, (0, pad_len_tgt), value=pad_token_id)
        padded_target_attention_mask = F.pad(target_attention_mask, (0, pad_len_tgt), value=0)

        # Make label for loss calculation
        target_len = padded_target_attention_mask.sum().item()
        target_labels = torch.full_like(padded_target_ids, -100)
        target_labels[target_len - a_len:target_len] = padded_target_ids[target_len - a_len:target_len]

        return {
            "doc_input_ids": torch.stack(padded_doc_ids),                   # (k, max_seq_length)
            "doc_attention_mask": torch.stack(padded_doc_masks),            # (k, max_seq_length)
            "source_input_ids": torch.stack(padded_source_ids),             # (k, max_seq_length)
            "source_attention_mask": torch.stack(padded_source_masks),      # (k, max_seq_length)
            "target_input_ids": padded_target_ids,                          # (k * max_seq_length + max_ans_length,)
            "target_attention_mask": padded_target_attention_mask,          # (k * max_seq_length + max_ans_length,)
            "doclen_list": torch.tensor(doclen_list),                       # (k,)
            "target_labels": target_labels,                                 # (k * max_seq_length + max_ans_length,)
            "a_len": torch.tensor(a_len)                                    # (1,)   
        }


class GGDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.batch_size = self.data_cfg.batch_size
        self.num_workers = self.data_cfg.num_workers

    def setup(self, stage: Optional[str] = None):
        full_data = load_relevance_dataset(self.data_cfg.data_path)
        if self.cfg.debug_mode:
            full_data = full_data[:5000]
        dataset = [asdict(item) for item in tqdm(full_data, desc="Converts to dict")]
        hf_dataset = HFDataset.from_list(dataset)
        split_dataset = hf_dataset.train_test_split(test_size=self.data_cfg.test_size, seed=self.data_cfg.seed)
        if stage == 'fit' or stage is None:
            self.train_dataset = GGDataset(split_dataset["train"], self.cfg)
            self.val_dataset = GGDataset(split_dataset["test"], self.cfg)
        elif stage == 'validate':
            self.val_dataset = GGDataset(split_dataset["test"], self.cfg)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
    def collate_fn(self, batch):
        """
        Collate function to combine multiple docs into a single batch.
        """
        B = len(batch)
        K, S_D = batch[0]["doc_input_ids"].shape
        flat_doc_input_ids = torch.stack([item["doc_input_ids"] for item in batch]).reshape(-1, S_D)  # (B * K, S_D)
        flat_doc_attention_mask = torch.stack([item["doc_attention_mask"] for item in batch]).reshape(-1, S_D)  # (B * K, S_D)
        flat_source_input_ids = torch.stack([item["source_input_ids"] for item in batch]).reshape(-1, S_D)  # (B * K, S_D)
        flat_source_attention_mask = torch.stack([item["source_attention_mask"] for item in batch]).reshape(-1, S_D)  # (B * K, S_D)
        
        return {
            "doc_input_ids": flat_doc_input_ids,                   # (B * K, S_D)
            "doc_attention_mask": flat_doc_attention_mask,        # (B * K, S_D)
            "source_input_ids": flat_source_input_ids,             # (B * K, S_D)
            "source_attention_mask": flat_source_attention_mask,  # (B * K, S_D)
            "target_input_ids": torch.stack([item["target_input_ids"] for item in batch]),          # (B, K * S_D + max_ans_length)
            "target_attention_mask": torch.stack([item["target_attention_mask"] for item in batch]),  # (B, K * S_D + max_ans_length)
            "doclen_list": torch.stack([item["doclen_list"] for item in batch]),                  # (B, K)
            "target_labels": torch.stack([item["target_labels"] for item in batch]),                   # (B, K * S_D + max_ans_length)
            "a_len": torch.stack([item["a_len"] for item in batch])                            # (B,)
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
    data_module = GGDataModule(cfg)
    data_module.setup("fit")
    for batch in data_module.train_dataloader():
        import pdb; pdb.set_trace()
        print(batch)
        break