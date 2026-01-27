from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import logging
import torch



@dataclass
class FeatureExample:
    features: torch.Tensor
    label: int

logger = logging.getLogger(__name__)

class ConflictDataset(Dataset):
    def __init__(self, data: List[FeatureExample]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        return {
            "features": item.features,
            "label": item.label
        }
    

class ConflictDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.train_dataset: Optional[ConflictDataset] = None
        self.val_dataset: Optional[ConflictDataset] = None

    def setup(self, stage: Optional[str] = None):
        dataset = torch.load(self.cfg.filepath, map_location="cpu", weights_only=False)
        train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        if stage == 'fit' or stage is None:
            self.train_dataset = ConflictDataset(train_dataset)
            self.val_dataset = ConflictDataset(valid_dataset)
        elif stage == 'validate':
            self.val_dataset = ConflictDataset(valid_dataset)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def collate_fn(self, batch):
        features = torch.stack([item['features'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return features, labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )