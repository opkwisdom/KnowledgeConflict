from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf, ListConfig
from transformers import AutoModelForCausalLM, AutoModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
import logging
import einops
import torch
import os
import h5py
import datetime
import omegaconf.base
from typing import Any
from torch.utils.data import Subset

from models import load_model
from datamodule import GGEmbDataModule
from utils import setup_logger, load_config

def setup_ddp():
    """DDP 환경 초기화 함수"""
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl",
                                timeout=datetime.timedelta(hours=5))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, int(os.environ["WORLD_SIZE"])
    else:
        # 단일 GPU 환경 대비 (Fallback)
        return 0, 1
    
def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def merge_h5_files(output_path, world_size):
    logger = logging.getLogger(__name__)
    logger.info("Merging temporary HDF5 files into the final output...")

    with h5py.File(output_path, "w") as final_h5:
        for rank in range(world_size):
            temp_path = f"{output_path}_rank{rank}.tmp"
            if not os.path.exists(temp_path):
                continue
            
            with h5py.File(temp_path, "r") as temp_h5:
                for group_name in tqdm(temp_h5.keys(), desc=f"Merging Rank {rank}"):
                    temp_h5.copy(group_name, final_h5)
            
            # os.remove(temp_path)
    logger.info(f"Successfully merged into {output_path}")


def get_model_name(model_name_or_path: str):
    if "llama" in model_name_or_path.lower():
        return "llama"
    elif "qwen" in model_name_or_path.lower():
        return "qwen"
    else:
        raise ValueError(f"Unsupported model name: {model_name_or_path}")


def prepare_modules(llm: AutoModelForCausalLM):
    for param in llm.parameters():
        param.requires_grad = False
    llm.enable_input_require_grads()
    llm.gradient_checkpointing_enable()
    llm.eval()

# TODO: Change the logic
def forward(llm: AutoModelForCausalLM, batch):
        """
        Forward pass to generate LLM hidden states for both documents and CA-Former input.
        Returns:
            doc_repr: Tensor of shape (B*K, D_llm) - LLM representation of the document (using <EOS> token)
            llm_repr: Tensor of shape (B*K, L, S, D_llm) - LLM hidden states for CA-Former input
        """
        # Generate LLM hidden states of documents, last layer only, <EOS> token
        # Since we use right-padding, we should track the position of <EOS> token
        with torch.no_grad():
            doc_outputs = llm(
                input_ids=batch["doc_input_ids"],
                attention_mask=batch["doc_attention_mask"],
                output_hidden_states=True,
            )
            last_hidden_state = doc_outputs.hidden_states[-1]  # (B*K, L, D_llm)
            last_token_indices = batch["doc_attention_mask"].sum(dim=-1) - 1   # (B*K,)
            D_llm = last_hidden_state.shape[-1]
            gather_indices = last_token_indices.view(-1, 1, 1).expand(-1, 1, D_llm)
            doc_repr = torch.gather(last_hidden_state, dim=1, index=gather_indices).squeeze(1)  # (B*K, D_llm)

        # Generate LLM hidden states for CA-Former input
        with torch.no_grad():
            llm_outputs = llm(
                input_ids=batch["source_input_ids"],
                attention_mask=batch["source_attention_mask"],
                output_hidden_states=True,
            )
            llm_repr = torch.stack(llm_outputs.hidden_states[-12:]).permute(1, 0, 2, 3)   # (B*K, L, S, D_llm)
        return doc_repr, llm_repr

# TODO: Change the logic
def compute_oracle_loss_gradients(llm: AutoModelForCausalLM, batch):
        """
        Compute the gradients of the oracle loss.
        Basically, we return the whole gradient matrix.
        Returns:
            loss_oracle: scalar tensor
            loss_gradients: Tensor of shape (B, k * max_seq_length + max_ans_length, D_llm)
        """
        inputs_embeds = llm.get_input_embeddings()(batch["target_input_ids"])
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        llm_outputs = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["target_attention_mask"],
            labels=batch["target_labels"]
        )
        loss_oracle = llm_outputs.loss
        del llm_outputs
        loss_gradients = torch.autograd.grad(
            outputs=loss_oracle,    # Starting point
            inputs=inputs_embeds,   # End point
            retain_graph=False,
            create_graph=False      # We don't need higher-order gradients here
        )[0]    # (B, k * max_seq_length + max_ans_length, D_llm)
        return loss_oracle.detach_(), loss_gradients

# TODO: Change the logic
def compute_oracle_scores(loss_gradients: torch.Tensor, doc_repr: torch.Tensor, batch: Any):
    """
    Compute the target scores for CA-Former based on the gradients of the oracle loss.
    Args:
        loss_gradients: Tensor of shape (B, k * max_seq_length + max_ans_length, D_llm)
        doc_repr: Tensor of shape (B*k, D_llm)
        doclen_list: Tensor of shape (B, k)
        a_len: Tensor of shape (B,)
        question_ids: Tensor of shape (B,)
    Returns:
        scores_oracle: List of length B. Each element is a Dict containing different
                       types of oracle scores for that specific batch item.
                       (All tensors are converted to numpy arrays for easier storage)
    """
    doclen_list = batch["doclen_list"]
    a_len = batch["a_len"]
    question_ids = batch["question_ids"]

    B = loss_gradients.shape[0]
    K = doc_repr.shape[0] // loss_gradients.shape[0]
    D = doc_repr.shape[-1]
    # Expand loss_gradients to match the shape of doc_repr
    loss_gradients_expanded = loss_gradients.unsqueeze(1).expand(-1, K, -1, -1)
    loss_gradients_expanded = loss_gradients_expanded.reshape(B*K, -1, D)
    neg_loss_gradients_expanded = -1 * loss_gradients_expanded
    
    modes = ["whole", "marginal", "marginal-qa", "topk", "topk-qa"]
    # ========== Whole score computation ==========
    # Most naive way in parallel, faster
    scores_oracle_whole = einops.einsum(doc_repr, neg_loss_gradients_expanded,
                                "b d, b l d -> b l").mean(dim=-1).reshape(B, -1)
    
    # ========== Marginal & Top-k score computation ==========
    # More fine-grained way using sequential processing, slower
    TOPK_RANGE = 10
    scores_oracle_marginal = torch.zeros(B, K, device=loss_gradients.device)
    scores_oracle_marginal_qa = torch.zeros(B, K, device=loss_gradients.device)
    scores_oracle_topk = torch.zeros(B, K, TOPK_RANGE, device=loss_gradients.device)
    scores_oracle_topk_qa = torch.zeros(B, K, TOPK_RANGE, device=loss_gradients.device)
    for i in range(B):
        start_pos = 0
        qa_len = a_len[i] + question_ids[i].shape[0]
        total_doc_len = doclen_list[i].sum()
        for j in range(K):
            flat_idx = i*K + j
            end_pos = start_pos + doclen_list[i, j]
            qa_gradients = neg_loss_gradients_expanded[flat_idx, total_doc_len:total_doc_len + qa_len]
            
            # ----- Marginal -----
            marginal_gradients = neg_loss_gradients_expanded[flat_idx, start_pos:end_pos]   # (doc_len, D_llm)
            each_doc_repr = doc_repr[flat_idx]     # (D_llm,)
            each_scores_oracle = einops.einsum(each_doc_repr, marginal_gradients,
                                                "d, l d -> l").mean(dim=-1)
            scores_oracle_marginal[i, j] = each_scores_oracle
            
            # ----- Marginal-QA -----
            marginal_qa_gradients = torch.cat([marginal_gradients, qa_gradients], dim=0)
            each_scores_oracle = einops.einsum(each_doc_repr, marginal_qa_gradients,
                                               "d, l d -> l").mean(dim=-1)
            scores_oracle_marginal_qa[i, j] = each_scores_oracle

            # ----- Top-k & Top-k-QA -----
            norms = marginal_gradients.norm(p=2, dim=-1)
            actual_topk = min(TOPK_RANGE, marginal_gradients.shape[0])
            if actual_topk > 0:
                topk_indices = torch.topk(norms, k=actual_topk).indices
                
                topk_grads = marginal_gradients[topk_indices]
                topk_scores = einops.einsum(each_doc_repr, topk_grads,
                                            "d, k d -> k")
                
                topk_qa_grads = torch.cat([topk_grads, qa_gradients], dim=0)
                topk_qa_scores = einops.einsum(each_doc_repr, topk_qa_grads,
                                               "d, k d -> k").mean(dim=-1)
                
                # Normalize
                range_vectors = torch.arange(1, actual_topk + 1, device=topk_scores.device)
                topk_cum_grads = torch.cumsum(topk_scores, dim=0) / range_vectors
                topk_qa_cum_grads = torch.cumsum(topk_qa_scores, dim=0) / range_vectors
                scores_oracle_topk[i, j, :actual_topk] = topk_cum_grads
                scores_oracle_topk_qa[i, j, :actual_topk] = topk_qa_cum_grads
            start_pos = end_pos
    
    # ========== Flatten to List of Dicts ==========
    scores_oracle = []
    for i in range(B):
        item_scores = {
            "whole": scores_oracle_whole[i].detach().cpu().float().numpy(),
            "marginal": scores_oracle_marginal[i].detach().cpu().float().numpy(),
            "marginal-qa": scores_oracle_marginal_qa[i].detach().cpu().float().numpy()
        }
        for topk in range(TOPK_RANGE):
            item_scores[f"topk_{topk+1}"] = scores_oracle_topk[i, :, topk].detach().cpu().float().numpy()
            item_scores[f"topk-qa_{topk+1}"] = scores_oracle_topk_qa[i, :, topk].detach().cpu().float().numpy()
        scores_oracle.append(item_scores)
    
    return scores_oracle

def get_completed_ids(world_size, final_output_path):
    completed = set()
    # import pdb; pdb.set_trace()
    for r in range(world_size):
        tmp_path = f"{final_output_path}_rank{r}.tmp"
        if os.path.exists(tmp_path):
            try:
                with h5py.File(tmp_path, "r") as h5file:
                    completed.update(set(h5file.keys()))
            except Exception as e:
                pass
    return completed

def get_filtered_subset(dataset, completed_ids, is_main_process=False):
    try:
        all_ids = dataset.data['idx']
        valid_indices = [i for i, uid in enumerate(all_ids) if str(uid) not in completed_ids]
    except Exception:
        valid_indices = []
        for i in tqdm(range(len(dataset)), desc="Filtering data", disable=not is_main_process):
            if str(dataset.data[i]["idx"]) not in completed_ids:
                valid_indices.append(i)
                
    return Subset(dataset, valid_indices)


def main():
    torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf, omegaconf.base.ContainerMetadata])
    # Allow TF32 (This can be useful for mixed precision training)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    local_rank, world_size = setup_ddp()
    is_main_process = (local_rank == 0)

    config = load_config()
    seed_everything(config.seed)
    experiment_name = "gen_precompute_table"
    config.output_dir = os.path.join(config.output_dir, experiment_name)

    if is_main_process:
        setup_logger("main", config.output_dir)
        logger = logging.getLogger(__name__)
        logger.info("Configuration Loaded:")
        logger.info(OmegaConf.to_yaml(config))
    else:
        logger = logging.getLogger(__name__)

    # Load datamodule & model
    datamodule = GGDataModule(config)
    # Prepare dataloaders
    if is_main_process:
        datamodule.prepare_data()
    if dist.is_initialized():
        dist.barrier()
    
    output_base_dir = os.path.join(config.output_dir, get_model_name(config.model.model_name))
    final_output_path = os.path.join(output_base_dir, "nq_val_precompute_table.h5")

    # Resume logic
    global_completed_ids = get_completed_ids(world_size, final_output_path)
    if is_main_process:
        logger.info(f"Found {len(global_completed_ids)} completed IDs from surviving files.")

    datamodule.setup(stage="fit")

    original_train_dataset = datamodule.train_dataloader().dataset
    original_val_dataset = datamodule.val_dataloader().dataset

    train_dataset = get_filtered_subset(original_train_dataset, global_completed_ids, is_main_process)
    val_dataset = get_filtered_subset(original_val_dataset, global_completed_ids, is_main_process)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=datamodule.collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=datamodule.collate_fn
    )

    llm, _ = load_model(config.model.model_name)
    llm = llm.to(local_rank)
    prepare_modules(llm)

    # Compute and save the precompute table
    # HDF5 format is chosen for better handling of large datasets and hierarchical storage
    # And it is a kind of virtual file system
    output_base_dir = os.path.join(config.output_dir, get_model_name(config.model.model_name))
    if is_main_process:
        os.makedirs(output_base_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    
    final_output_path = os.path.join(output_base_dir, "nq_val_precompute_table.h5")
    temp_output_path = f"{final_output_path}_rank{local_rank}.tmp"

    with h5py.File(temp_output_path, "a") as h5file:
        for _split, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:
            logger.info(f"Processing {_split} split...")
            iterator = tqdm(dataloader, desc=f"[Rank {local_rank}] Processing {_split}", position=local_rank)

            for batch in iterator:
                batch_ids = [str(idx.item()) for idx in batch["idx"]]
                if all(b_id in h5file for b_id in batch_ids):
                    continue

                batch = {k: v.to(local_rank) if hasattr(v, "to") else v for k, v in batch.items()}
                doc_repr, llm_repr = forward(llm, batch)
                del llm_repr

                with torch.enable_grad():
                    loss_oracle, loss_gradients = compute_oracle_loss_gradients(llm, batch)
                    scores_oracle = compute_oracle_scores(loss_gradients, doc_repr, batch)
                
                # Store the gradients in the precompute table
                for i in range(batch["idx"].shape[0]):
                    example_id = str(batch["idx"][i].item())

                    if example_id in h5file:
                        continue
                
                    grp = h5file.create_group(example_id)
                    for score_type, score_values in scores_oracle[i].items():
                        grp.create_dataset(score_type, data=score_values)

    if dist.is_initialized():
        dist.barrier()
    if is_main_process:
        merge_h5_files(final_output_path, world_size)
    cleanup_ddp()


if __name__ == "__main__":
    main()