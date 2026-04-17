from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset   # Prevent OOM
from typing import List, Optional
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import logging
import torch

from utils import load_collection

logger = logging.getLogger(__name__)


class CTDataset(Dataset):
    def __init__(self, data: HFDataset, cfg: DictConfig):
        self.data = data
        self.cfg = cfg
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.roberta_tokenizer.padding_side = "right"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token_id = 128004
        self.llm_tokenizer.padding_side = "left"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        context = f"Title: {item['title']}\n\nText: {item['text']}"
        roberta_inputs = self.roberta_tokenizer(
            context,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.max_seq_length,
            return_tensors="pt",
        )
        llm_inputs = self.llm_tokenizer(
            context,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.max_seq_length,
            return_tensors="pt",
        )
        return {"roberta": roberta_inputs, "llm": llm_inputs}


class CTDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.batch_size = self.data_cfg.batch_size
        self.num_workers = self.data_cfg.num_workers
    
    def setup(self, stage: Optional[str] = None):
        full_data = load_collection(self.data_cfg.data_path)
        hf_dataset = HFDataset.from_list(full_data)
        split_dataset = hf_dataset.train_test_split(test_size=self.data_cfg.test_size, seed=self.data_cfg.seed)
        # train_data, val_data = train_test_split(full_data, test_size=self.data_cfg.test_size, random_state=self.data_cfg.seed)
        if stage == 'fit' or stage is None:
            self.train_dataset = CTDataset(split_dataset["train"], self.cfg)
            self.val_dataset = CTDataset(split_dataset["test"], self.cfg)
        elif stage == 'validate':
            self.val_dataset = CTDataset(split_dataset["test"], self.cfg)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
    def collate_fn(self, batch):
        roberta_inputs = {
            key: torch.cat([item["roberta"][key] for item in batch], dim=0)
            for key in batch[0]["roberta"].keys()
        }
        llm_inputs = {
            key: torch.cat([item["llm"][key] for item in batch], dim=0)
            for key in batch[0]["llm"].keys()
        }
        return {"roberta": roberta_inputs, "llm": llm_inputs}
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,  # Ensure consistent batch sizes for multi-GPU training
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,  # Ensure consistent batch sizes for multi-GPU training
        )


# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     cfg = OmegaConf.load("config/src/train/run_ct_train.yaml")
#     datamodule = CTDataModule(cfg)
#     datamodule.setup("fit")
#     for batch in datamodule.train_dataloader():
#         import pdb; pdb.set_trace()
#         print(batch)
#         break