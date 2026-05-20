from pytorch_lightning import seed_everything
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import List
from tqdm import tqdm
from datetime import timedelta
from sentence_transformers import CrossEncoder
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from dataclasses import dataclass, asdict
import torch.distributed as dist
import logging
import torch
import os
import re
import glob
import json

from models import MultiHiddenCAFormerForGG, CAFormerGGClassifier, DISCA, HybridDISCA
from utils import (
    setup_logger, load_config, load_qa_dataset, compute_metrics, validate_and_save_results,
    QAExample, InferenceResult, MetricResult
)

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2),
            device_id=device
        )
    
    return local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def load_checkpoint(model: CAFormerGGClassifier, checkpoint_dir):
    logger = logging.getLogger(__name__)
    if checkpoint_dir is None:
        logger.info(f"No checkpoint path specified.")
        return model, False

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpt_files:
        logger.info(f"No checkpoint path specified.")
        return model, False
    
    pattern = re.compile(r'valid_loss=(-?\d+\.?\d*)')
    candidates = []
    for path in ckpt_files:
        filename = os.path.basename(path)
        match = pattern.search(filename)
        if match:
            value = float(match.group(1))
            candidates.append((value, path))

    best = min(candidates, key=lambda x: x[0])
    logger.info(f"Loading CAFormer checkpoint from {best[1]}...")
    checkpoint = torch.load(best[1], map_location="cpu", weights_only=False)
    
    cleaned_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("caformer_clf."):
            new_key = key[len("caformer_clf."):]
            cleaned_state_dict[new_key] = value
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    logger.info(f"Missing keys: {missing_keys}")
    logger.info(f"Unexpected keys: {unexpected_keys}")

    return model, True




class QADataset(Dataset):
    def __init__(self, examples, first_topn=50):
        self.examples = examples
        self.first_topn = first_topn

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "idx": idx,                       # 글로벌 원본 인덱스
            "question": item.question,
            "ctxs": item.ctxs[:self.first_topn],
            "answers": item.answers,
        }

def collate_fn(batch):
    return {
        "idx": [b["idx"] for b in batch],
        "questions": [b["question"] for b in batch],
        "contexts_list": [b["ctxs"] for b in batch],
        "answers": [b["answers"] for b in batch],
    }

def run_inference_ddp(
    config: DictConfig,
    model: DISCA,
    dataset: List[QAExample],
    rank: int,
    world_size: int,
) -> List[InferenceResult]:
    """각 rank가 자신의 shard만 처리. 결과는 idx와 함께 반환되어 나중에 정렬 가능."""
    qa_dataset = QADataset(dataset)
    sampler = DistributedSampler(
        qa_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False
    )
    dataloader = DataLoader(
        qa_dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    outputs = []
    pbar = tqdm(
        dataloader,
        desc=f"[Rank {rank}] Inference",
        position=rank,
    )
    for batch in pbar:
        queries = batch["questions"]
        contexts_list = batch["contexts_list"]
        gold_answers_list = batch["answers"]
        global_idxs = batch["idx"]

        batch_answers = model.generate(queries, contexts_list)

        for gidx, query, pred_answer, gold_answers in zip(
            global_idxs, queries, batch_answers, gold_answers_list
        ):
            metrics = compute_metrics(pred_answer, gold_answers)
            result = InferenceResult(
                id=gidx,
                question=query,
                pred_answer=pred_answer,
                answers=gold_answers,
                metrics=metrics,
            )
            outputs.append(result)
    return outputs

def save_local_results(local_results, output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"_results_rank_{rank}.json")
    with open(path, "w") as f:
        json.dump([asdict(r) for r in local_results], f)
    return path

def gather_from_files(output_dir, world_size):
    merged = []
    for rank in range(world_size):
        path = os.path.join(output_dir, f"_results_rank_{rank}.json")
        with open(path) as f:
            chunk = json.load(f)
        for d in chunk:
            d["metrics"] = MetricResult(**d["metrics"])
            merged.append(InferenceResult(**d))
    
    merged.sort(key=lambda r: r.id)
    seen = set()
    deduped = []
    for r in merged:
        if r.id not in seen:
            seen.add(r.id)
            deduped.append(r)

    for rank in range(world_size):
        path = os.path.join(output_dir, f"_results_rank_{rank}.json")
        if os.path.exists(path):
            os.remove(path)
    
    return deduped


def main():
    torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf])
    
    # === DDP setup ===
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    config = load_config()
    seed_everything(config.seed)
    output_dir = os.path.join(config.output_dir, config.model.model_name.split('/')[-1], config.data.name)
    config.output_dir = os.path.join(output_dir, config.experiment_name)
    if is_main_process():
        setup_logger("main", config.output_dir)
        logger = logging.getLogger(__name__)
        logger.info("Configuration Loaded:")
        logger.info(OmegaConf.to_yaml(config))
        logger.info(f"DDP world size={world_size}")
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

    # Load model & dataset
    config.caformer.llm_width = 4096  # post-init
    caformer = MultiHiddenCAFormerForGG(config.caformer).to(
        device=f"cuda:{local_rank}", dtype=torch.bfloat16)
    caformer_clf = CAFormerGGClassifier(config, caformer).to(
        device=f"cuda:{local_rank}", dtype=torch.bfloat16)
    # Load CAFormerGGClassifier weights from the best checkpoint of stage 3
    caformer_clf, load_success = load_checkpoint(caformer_clf, config.caformer.ckpt_dir)

    if not load_success:
        if is_main_process():
            logger.error("Failed to load model checkpoint. Exiting inference.")
        cleanup_ddp()
        return

    if not config.do_hybrid:
        model = DISCA(config, config.model.model_name, caformer_clf)
        logger.info("Initialized DISCA model without reranker.")
    else:
        ce_reranker = CrossEncoder(config.ce_reranker_path).to(device=f"cuda:{local_rank}")
        model = HybridDISCA(config, config.model.model_name, caformer_clf, ce_reranker)
        logger.info(f"Initialized HybridDISCA model with CE reranker {config.ce_reranker_path}.")

    dataset = load_qa_dataset(config.data.data_path)

    # Do inference on validation set and save results
    local_results = run_inference_ddp(config, model, dataset, rank, world_size)
    save_local_results(local_results, config.output_dir, rank)

    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    print(f"[Rank {rank}] Saved local results", flush=True)

    if is_main_process():
        all_results = gather_from_files(config.output_dir, world_size)
        validate_and_save_results(all_results, config.output_dir, logger)
        logger.info(f"Saved {len(all_results)} results to {config.output_dir}")

    dist.barrier(device_ids=[local_rank])
    print(f"[Rank {rank}] Before cleanup", flush=True)
    cleanup_ddp()
    print(f"[Rank {rank}] Done", flush=True)

if __name__ == "__main__":
    main()