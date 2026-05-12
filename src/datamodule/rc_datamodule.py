from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, load_from_disk   # Prevent OOM
from typing import List, Optional
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
import logging
import torch
import os

from utils import load_collection

logger = logging.getLogger(__name__)


class RCDataset(Dataset):
    def __init__(self, data: HFDataset, cfg: DictConfig):
        self.data = data
        self.cfg = cfg
        self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token_id = 128004
        self.llm_tokenizer.padding_side = "left"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        context = f"Title: {item['title']}\n\nText: {item['text']}"
        full_text = f"Reconstruct the given context\n\n{context}"

        tokenized = self.llm_tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.max_seq_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0)
        }


class RCDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.batch_size = self.data_cfg.batch_size
        self.num_workers = self.data_cfg.num_workers
    
    def prepare_data(self):
        cache_path = f"{self.data_cfg.data_path}_hf_cache"
        if not os.path.exists(cache_path):
            full_data = load_collection(self.data_cfg.data_path)
            hf_dataset = HFDataset.from_list(full_data)
            hf_dataset.save_to_disk(cache_path)
    
    def setup(self, stage: Optional[str] = None):
        # full_data = load_collection(self.data_cfg.data_path)
        # hf_dataset = HFDataset.from_list(full_data)
        cache_path = f"{self.data_cfg.data_path}_hf_cache"
        hf_dataset = load_from_disk(cache_path)
        split_dataset = hf_dataset.train_test_split(test_size=self.data_cfg.test_size, seed=self.data_cfg.seed)
        if stage == 'fit' or stage is None:
            self.train_dataset = RCDataset(split_dataset["train"], self.cfg)
            self.val_dataset = RCDataset(split_dataset["test"], self.cfg)
        elif stage == 'validate':
            self.val_dataset = RCDataset(split_dataset["test"], self.cfg)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/src/train/run_rc_train.yaml")
    data_module = RCDataModule(cfg)
    data_module.setup("fit")
    for batch in data_module.train_dataloader():
        import pdb; pdb.set_trace()
        print(batch)
        break