from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
import logging
import torch

from utils import load_relevance_dataset


RELEVANCE_MAPPING = {
    "positive": 0,
    "supportive": 0,
    "negative": 1,
    "contradictory": 1,
    "irrelevant": 2
}

logger = logging.getLogger(__name__)

class CADataset(Dataset):
    def __init__(self, data: List[torch.Tensor], cfg: DictConfig):
        self.data = data
        self.cfg = cfg
        self.topk_per_query = getattr(cfg, "topk_per_query", None) or 10
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        query = item.question
        ctxs = item.ctxs[:self.topk_per_query]
        relevance_mapping = item.ctx_relevance.mapping
        labels = [RELEVANCE_MAPPING.get(v, "irrelevant") for k, v \
                  in list(relevance_mapping.items())[:self.topk_per_query]]
        if len(labels) < self.topk_per_query:
            labels.extend([2] * (self.topk_per_query - len(labels)))
        return query, ctxs, labels
    

class CADataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
    
    def setup(self, stage: Optional[str] = None):
        full_data = load_relevance_dataset(self.cfg.data_path)
        train_data, val_data = train_test_split(full_data, test_size=self.cfg.test_size, random_state=self.cfg.seed)
        if stage == 'fit' or stage is None:
            self.train_dataset = CADataset(train_data, self.cfg)
            self.val_dataset = CADataset(val_data, self.cfg)
        elif stage == 'validate':
            self.val_dataset = CADataset(val_data, self.cfg)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def collate_fn(self, batch):
        queries, ctxs_list, labels_list = zip(*batch)
        return {
            "queries": list(queries),
            "ctxs_list": list(ctxs_list),
            "labels_tensor": torch.tensor(list(labels_list), dtype=torch.long).reshape(-1)  # (B*N_{docs},)
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            shuffle=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )