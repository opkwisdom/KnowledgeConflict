from pytorch_lightning import seed_everything
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import List
from tqdm import tqdm
from dataclasses import dataclass, asdict
import logging
import torch
import os
import json

from models import MultiHiddenCAFormer, CAFormerGGClassifier, DISCA, load_model
from utils import (
    setup_logger, load_config, load_qa_dataset, compute_metrics,
    RelevanceQAExample, InferenceResult
)

def load_checkpoint(model: CAFormerGGClassifier, checkpoint_path):
    logger = logging.getLogger(__name__)

    if not os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint not found at {checkpoint_path}. Skipping checkpoint loading.")
        return model, False
    logger.info(f"Loading CAFormerGGClassifier weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
    dataset: List[RelevanceQAExample]
) -> List[InferenceResult]:
    logger = logging.getLogger(__name__)
    outputs = []
    batch_size = config.data.batch_size
    for i in tqdm(range(0, len(dataset), batch_size), desc="Running Inference"):
        batch = dataset[i:i+batch_size]
        queries = [item.question for item in batch]
        contexts_list = [item.ctxs for item in batch]
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



def save_results(
    inference_list: List[InferenceResult],
    output_dir: str
) -> None:
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = f"{output_dir}/inference_summary.txt"
    all_results_path = f"{output_dir}/inference_results.json"

    total = len(inference_list)
    correct = sum([1 for res in inference_list if res.metrics.soft_em])
    recall = sum([res.metrics.recall for res in inference_list]) / total if total > 0 else 0.0
    precision = sum([res.metrics.precision for res in inference_list]) / total if total > 0 else 0.0
    f1 = sum([res.metrics.f1 for res in inference_list]) / total if total > 0 else 0.0

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Total={total}, Correct={correct}, Accuracy={accuracy:.4f},"
                f" Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved inference summary to {summary_path}")
    with open(all_results_path, 'w') as f:
        json_results = [asdict(res) for res in inference_list]
        json.dump(json_results, f, ensure_ascii=False, indent=4)


def main():
    torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf])
    
    config = load_config()
    seed_everything(config.seed)
    config.output_dir = os.path.join(config.output_dir, config.experiment_name)
    setup_logger("main", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load model & dataset
    config.caformer.llm_width = 4096  # post-init
    caformer = MultiHiddenCAFormer(config.caformer).to(dtype=torch.bfloat16)
    caformer_clf = CAFormerGGClassifier(config, caformer).to(dtype=torch.bfloat16)
    # Load CAFormerGGClassifier weights from the best checkpoint of stage 3
    caformer_clf, load_success = load_checkpoint(caformer_clf, config.caformer.ckpt_path)

    model = DISCA(config, config.model.model_name, caformer_clf)

    if not load_success:
        logger.error("Failed to load model checkpoint. Exiting inference.")
        return
    dataset = load_qa_dataset(config.data.data_path)
    dataset = dataset[:10]  # For quick testing

    # Do inference on validation set and save results
    results = run_inference(config, model, dataset)
    save_results(results, config.output_dir)

if __name__ == "__main__":
    main()