import faiss
import logging
import numpy as np
import os
import pickle
from tqdm import tqdm
import time
from omegaconf import DictConfig

from typing import List, Tuple, Dict

logger = logging.getLogger()

class FaissIndexer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.use_gpu = getattr(cfg, "use_gpu", False)
        
        self.index_type = cfg.index_type
        self.vector_dim = cfg.vector_dim
        self.index_id_to_db_id = []
        self.index = self._create_cpu_index(self.index_type, self.vector_dim)
        if self.use_gpu:
            self.index = self._to_gpu(self.index, cfg)

    def _create_cpu_index(self, index_type: str, dim: int):
        if index_type == "flat":
            return faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.cfg.M)  # the number of neighbors
            index.hnsw.efConstruction = self.cfg.ef_construction
            index.hnsw.efSearch = self.cfg.ef_search
            return index
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
    def _to_gpu(self, cpu_index: faiss.Index, cfg: DictConfig):
        """
        Convert cpu-index to gpu-index, and configure gpu index options
        """
        co = faiss.GpuMultipleClonerOptions()
        co.shard = getattr(cfg, "shard", False)
        return faiss.index_cpu_to_all_gpus(cpu_index, co=co)
        
    def index_data(self, ids: List[str], vectors: np.ndarray, buffer_size: int = 50000):
        """
        Index the provided vectors with corresponding ids.
        Args:
            ids: List of doc ids.
            vectors: (N, D) numpy array of vectors.
        """
        n = vectors.shape[0]
        assert len(ids) == n, "Number of ids must match number of vectors"
        
        iterator = tqdm(range(0, n, buffer_size), desc="Indexing vectors")
        s = time.time()
        for start_idx in iterator:
            end_idx = min(start_idx + buffer_size, n)
            batch_vectors = vectors[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            self.index.add(batch_vectors)
            self.index_id_to_db_id.extend(batch_ids)
        e = time.time()
        logger.info(f"Indexed {n} vectors: {e-s:.2f} sec")
    
    def search(self, query_vectors: np.ndarray, query_indices: List[str], top_k: int, batch_size: int = 128) -> Dict[str, List[str]]:
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)

        logger.info(f"Searching for top {top_k} nearest neighbors...")
        
        n = query_vectors.shape[0]
        iterator = tqdm(range(0, n, batch_size), desc="Searching queries")
        
        total_indices = []
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n)
            batch_queries = query_vectors[start_idx:end_idx]
            _, indices = self.index.search(batch_queries, top_k)
            total_indices.append(indices)
        total_indices = np.vstack(total_indices)  # (N, top_k)
        
        # mapping faiss internal ids to db ids
        mapped_ids = {}
        for query_idx, topk_indices in zip(query_indices, total_indices):
            topk_mapped = [self.index_id_to_db_id[idx] if idx != -1 else None for idx in topk_indices]
            mapped_ids[query_idx] = topk_mapped

        return mapped_ids  # (N, top_k)
    
    def save(self, path: str):
        index_file = path + ".index.faiss"
        meta_file = path + ".index_meta.pkl"

        logger.info(f"Saving index to {index_file}...")
        if faiss.get_num_gpus() > 0 and self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_file)
        else:
            faiss.write_index(self.index, index_file)

        with open(meta_file, "wb") as f:
            pickle.dump(self.index_id_to_db_id, f)
        logger.info("Save completed.")

    def load(self, path: str):
        index_file = path + ".index.faiss"
        meta_file = path + ".index_meta.pkl"

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"No index file found at {index_file}")
        
        logger.info(f"Loading index from {index_file}...")
        self.index = faiss.read_index(index_file)

        with open(meta_file, "rb") as f:
            self.index_id_to_db_id = pickle.load(f)
        logger.info(f"Load completed. Total size: {self.index.ntotal}")