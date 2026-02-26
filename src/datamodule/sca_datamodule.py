from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import logging
import torch


class SCADataset(Dataset):
    def __init__(self, data: List[torch.Tensor]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        return
    

class SCADataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
    
    def setup(self, stage: Optional[str] = None):
        # Load your dataset here and split into train/val/test
        # For example:
        # full_data = load_your_data_function(self.cfg.data_path)
        # train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42)
        if stage == 'fit' or stage is None:
            pass
        elif stage == 'validate':
            pass
        else:
            raise ValueError(f"Unknown stage: {stage}")
        # self.train_dataset = SCADataset(train_data)
        # self.val_dataset = SCADataset(val_data)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
