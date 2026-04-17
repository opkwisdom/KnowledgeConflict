from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
from dataclasses import dataclass, asdict
from pydantic import BaseModel
from tqdm import tqdm
import logging
import json
import os

from KVzip.model import ModelKVzip
from models import DISCA, SingleHiddenCAFormer, CAFormerClassifier
from prompt import ALL_PROMPTS, PSEUDO_PASSAGE_PROMPT, GENERATE_PROMPT
from utils import (
    setup_logger, load_config, load_relevance_dataset, compute_metrics,
    MetricResult, RelevanceQAExample
)

def test_sca_pipeline(
    config: DictConfig,
    disca: DISCA,
    data: List[RelevanceQAExample]
) -> Tuple[List[torch.FloatTensor], List[torch.LongTensor]]:
    logger = logging.getLogger(__name__)

    batch_size = config.data.batch_size
    topk_per_query = config.data.topk_per_query

    caformer_input_list = []
    caformer_mask_list = []
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    for batch in tqdm(batches, desc="Testing SCA Pipeline", total=len(batches)):
        queries = [item.question for item in batch]
        contexts_list = [item.ctxs[:topk_per_query] for item in batch]
        caformer_input, caformer_mask = disca(queries, contexts_list)
        
        caformer_input_list.append(caformer_input.cpu())
        caformer_mask_list.append(caformer_mask.cpu())

    logger.info(f"SCA pipeline test completed. Processed {len(data)} examples in batches of {batch_size}.")
    return caformer_input_list, caformer_mask_list


def test_single_hidden_caformer(
    config: DictConfig,
    caformer: SingleHiddenCAFormer,
    caformer_input_list: List[torch.FloatTensor],
    caformer_mask_list: List[torch.LongTensor]
) -> List[torch.FloatTensor]:
    logger = logging.getLogger(__name__)
    all_outputs = []
    for caformer_input, caformer_mask in tqdm(zip(caformer_input_list, caformer_mask_list),
                                              desc="Testing SingleHiddenCAformer", total=len(caformer_input_list)):
        outputs = caformer(caformer_input, caformer_mask)
        all_outputs.append(outputs.cpu())
    logger.info(f"SingleHiddenCAformer test completed. Processed {len(caformer_input_list)} batches.")
    return all_outputs


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, T: float = 1.0):
        super().__init__()
        self.T = T
    
    def forward(self, input: torch.FloatTensor, target: torch.LongTensor):
        """
        Args:
            input: Tensor of shape (B, D)
            target: Tensor of shape (B,), where each value is in {0, 1, 2}
                    representing positive, negative, irrelevant
        Returns:
            loss: Scalar tensor representing the contrastive loss
        """
        input = F.normalize(input, p=2, dim=1)
        batch_scores = torch.exp(torch.matmul(input, input.T) / self.T) # (B, B)
        base_mask = torch.ones_like(batch_scores) \
            - torch.eye(batch_scores.size(0), device=batch_scores.device) # exclude self-similarity
        group_mask = (target.unsqueeze(1) == target.unsqueeze(0))   # S/C/I group mask
        batch_mask = base_mask * group_mask.float()

        denom = torch.sum(batch_scores * base_mask, dim=1, keepdim=True)    # (B, 1)
        log_prob = torch.log(batch_scores / (denom + 1e-9))
        num_positives = torch.sum(batch_mask, dim=1)   # (B,)

        loss = -torch.sum(log_prob * batch_mask, dim=1) / (num_positives + 1e-9)   # (B,)
        loss = loss.mean()
        return loss


def test_caformer_classifier(
    config: DictConfig,
    classifier: CAFormerClassifier,
    data: List[RelevanceQAExample],
    caformer_input_list: List[torch.FloatTensor],
    caformer_mask_list: List[torch.LongTensor]
):
    logger = logging.getLogger(__name__)
    all_logits = []

    label_mapping = {
        "positive": 0,
        "negative": 1,
        "irrelevant": 2
    }

    cls_loss_fn = torch.nn.CrossEntropyLoss()
    ctr_loss_fn = ContrastiveLoss()
    ctr_loss_weight = getattr(config.caformer, "ctr_loss_weight", 1.0)

    batches = [data[i:i + config.data.batch_size] for i in range(0, len(data), config.data.batch_size)]
    for batch, caformer_input, caformer_mask in tqdm(zip(batches, caformer_input_list, caformer_mask_list),
                                              desc="Testing CAFormerClassifier", total=len(caformer_input_list)):
        logits, pooled_output = classifier(caformer_input, caformer_mask)

        ### TODO: Compute loss for this batch
        # Loss = Classification loss (query-dependent) + Contrastive loss (query-agnostic)
        batch_ctx_mapping = [item.ctx_relevance.mapping for item in batch]
        batch_labels = []
        for mapping in batch_ctx_mapping:
            for idx in range(config.data.topk_per_query):
                label = mapping.get(idx, "irrelevant")
                batch_labels.append(label_mapping[label])
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(logits.device)
        
        ce_loss = cls_loss_fn(logits, batch_labels)
        ctr_loss = ctr_loss_fn(pooled_output, batch_labels)
        loss = ce_loss + ctr_loss_weight * ctr_loss

        all_logits.append(logits.cpu())
        logger.info(f"Batch Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, CTR: {ctr_loss.item():.4f})")
        
    logger.info(f"Testing CAFormerClassifier on {len(data)} examples.")
    return all_logits



def main():
    config = load_config()
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = "disca_test"
    config.output_dir = os.path.join(config.output_dir, experiment_name, cur_time)
    setup_logger(f"main_{cur_time}", config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(config))

    # Load data
    data = load_relevance_dataset(config.data.data_path)
    data = data[:config.data.batch_size * 8]  # Only load a small subset for testing
    logger.info(f"Loaded {len(data)} examples from {config.data.data_path}")

    # Initialize model
    repeat_prompt = ALL_PROMPTS[config.self_task_prompt_name]
    base_prompt = GENERATE_PROMPT[config.base_prompt_name]
    generate_prompt = GENERATE_PROMPT[config.generate_prompt_name]
    logger.info(f"Using repeat prompt: {repeat_prompt}")
    logger.info(f"Using base prompt: {base_prompt}")
    logger.info(f"Using generate prompt: {generate_prompt}")

    kvzip = ModelKVzip(config.model.model_name, gen_kwargs=config.model.gen_kwargs, prompt=repeat_prompt)
    logger.info(f"Model {config.model.model_name} initialized.")
    disca = DISCA(config, kvzip, generate_prompt, base_prompt)
    llm_tokenizer = disca.tokenizer
    logger.info("DISCA model initialized.")

    # Run test
    ##### ===== First test ===== #####
    caformer_input_list, caformer_mask_list = test_sca_pipeline(config, disca, data)
    
    ##### ===== Second test ===== #####
    config.caformer.llm_width = disca.model.config.hidden_size
    caformer = SingleHiddenCAFormer(config.caformer, llm_tokenizer).to(device=disca.device, dtype=torch.bfloat16)
    caformer_outputs = test_single_hidden_caformer(config, caformer, caformer_input_list, caformer_mask_list)
    del caformer_outputs    # Too large to keep in memory
    
    ##### ===== Third test ===== #####
    classifier = CAFormerClassifier(config, caformer).to(device=disca.device, dtype=torch.bfloat16)
    classifier_outputs = test_caformer_classifier(config, classifier, data, caformer_input_list, caformer_mask_list)

if __name__ == "__main__":
    main()
