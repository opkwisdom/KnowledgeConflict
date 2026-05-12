from pytorch_lightning import seed_everything
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import List
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from dataclasses import dataclass, asdict
import torch.distributed as dist
import logging
import torch
import os
import re
import glob
import json

from models import MultiHiddenCAFormerForGG, CAFormerGGClassifier, DISCA, load_model
from utils import (
    setup_logger, load_config, load_qa_dataset, compute_metrics, validate_and_save_results,
    QAExample, InferenceResult
)

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
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


def run_inference(
    config: DictConfig,
    model: DISCA,
    dataset: List[QAExample]
) -> List[InferenceResult]:
    outputs = []
    batch_size = config.data.batch_size
    for i in tqdm(range(0, len(dataset), batch_size), desc="Running Inference"):
        batch = dataset[i:i+batch_size]
        queries = [item.question for item in batch]
        contexts_list = [item.ctxs[10:60] for item in batch]  # exclude the top 10 gold
        answers = [item.answers for item in batch]
        
        batch_answers = model.generate(queries, contexts_list)
        for idx, (query, pred_answer, gold_answers) in enumerate(zip(queries, batch_answers, answers)):
            metrics = compute_metrics(pred_answer, gold_answers)
            result = InferenceResult(
                id=i+idx,
                question=query,
                pred_answer=pred_answer,
                answers=gold_answers,
                metrics=metrics
            )
            outputs.append(result)
    return outputs






class QADataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "idx": idx,                       # 글로벌 원본 인덱스
            "question": item.question,
            "ctxs": item.ctxs[10:60],         # exclude top 10 gold
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

def gather_results(local_results, world_size):
    """모든 rank의 결과를 rank 0으로 모음."""
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_results)
    if is_main_process():
        merged = []
        for chunk in gathered:
            merged.extend(chunk)
        # 글로벌 인덱스 기준 정렬해서 원본 데이터 순서 복원
        merged.sort(key=lambda r: r.id)
        seen = set()
        deduped = []
        for r in merged:
            if r.id not in seen:
                seen.add(r.id)
                deduped.append(r)
        return deduped
    return None


def main():
    torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf])
    
    # === DDP setup ===
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    config = load_config()
    seed_everything(config.seed)
    config.output_dir = os.path.join(config.output_dir, config.experiment_name)
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

    model = DISCA(config, config.model.model_name, caformer_clf)

    dataset = load_qa_dataset(config.data.data_path)
    # dataset = dataset[:10]  # For quick testing

    # Do inference on validation set and save results
    # results = run_inference(config, model, dataset)
    local_results = run_inference_ddp(config, model, dataset, rank, world_size)
    dist.barrier()
    all_results = gather_results(local_results, world_size)

    if is_main_process():
        validate_and_save_results(all_results, config.output_dir, logger)
        logger.info(f"Saved {len(all_results)} results to {config.output_dir}")

    cleanup_ddp()

if __name__ == "__main__":
    main()