from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf, ListConfig
from transformers import AutoModelForCausalLM, AutoModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
import torch.nn as nn
import logging
import einops
import torch
import os
import h5py
import glob
import datetime
import omegaconf.base
from torch.utils.data import Subset

from models import load_model
from datamodule import GGDataModule, GGChunkDataModule
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
    logger.info("Merging chunked temporary HDF5 files into the final output...")

    search_pattern = f"{output_path}_rank*_part*.tmp"
    temp_files = glob.glob(search_pattern)
    
    if not temp_files:
        logger.warning(f"No temporary files found matching: {search_pattern}")
        return
    
    temp_files.sort()
    
    with h5py.File(output_path, "w") as final_h5:
        for temp_path in temp_files:
            file_basename = os.path.basename(temp_path)
            with h5py.File(temp_path, "r") as temp_h5:
                for group_name in tqdm(temp_h5.keys(), desc=f"Merging {file_basename}"):
                    if group_name in final_h5:
                        temp_h5.copy(group_name, final_h5)

            # os.remove(temp_path)
    logger.info(f"Successfully merged {len(temp_files)} temporary files into {output_path}")


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
    if hasattr(llm.config, "attention_dropout"):
        llm.config.attention_dropout = 0.0
    if hasattr(llm.config, "dropout"):
        llm.config.dropout = 0.0
    # llm.eval()

def forward(llm: AutoModelForCausalLM, batch):
    """
    Forward pass to generate LLM hidden states for both documents and CA-Former input.
    Returns:
        doc_repr: Tensor of shape (B*TOTAL_K, D_llm) - LLM representation of the document (using <EOS> token)
    """
    # Generate LLM hidden states of documents, last layer only, <EOS> token
    # Since we use right-padding, we should track the position of <EOS> token
    doc_ids = batch["doc_input_ids"]
    doc_mask = batch["doc_attention_mask"]
    
    total_samples = doc_ids.shape[0]
    chunk_size = 10
    
    doc_repr_list = []
    llm_repr_list = []
    
    with torch.no_grad():
        for i in range(0, total_samples, chunk_size):
            # doc repr
            mini_doc_ids = doc_ids[i : i + chunk_size]
            mini_doc_mask = doc_mask[i : i + chunk_size]
            doc_outputs = llm(
                input_ids=mini_doc_ids,
                attention_mask=mini_doc_mask,
                output_hidden_states=True,
            )
            last_hidden_state = doc_outputs.hidden_states[-1]  # (B*K, L, D_llm)
            last_token_indices = mini_doc_mask.sum(dim=-1) - 1   # (B*K,)
            D_llm = last_hidden_state.shape[-1]
            gather_indices = last_token_indices.view(-1, 1, 1).expand(-1, 1, D_llm)
            mini_doc_repr = torch.gather(last_hidden_state, dim=1, index=gather_indices).squeeze(1)  # (B*K, D_llm)

            doc_repr_list.append(mini_doc_repr)

    doc_repr = torch.cat(doc_repr_list, dim=0)  # (B*TOTAL_K, D_llm)
    return doc_repr

def compute_oracle_loss_gradients(llm: AutoModelForCausalLM, batch):
    """
    Compute the gradients of the oracle loss for chunked inputs.
    Basically, we return the whole gradient matrix.
    Returns:
        loss_oracle: scalar tensor
        loss_gradients: Tensor of shape (B, C, L_T, D_llm)
    """
    B, C, L_T = batch["target_input_ids"].shape
    D_llm = llm.config.hidden_size
    
    # Enable gradient tracking
    full_target_ids = batch["target_input_ids"] # (B, C, L_T)
    full_inputs_embeds = llm.get_input_embeddings()(full_target_ids)    # (B, C, L_T, D_llm)
    full_inputs_embeds = full_inputs_embeds.detach().requires_grad_(True)
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    all_gradients = torch.zeros(B, C, L_T, D_llm, device=full_inputs_embeds.device)
    total_loss_val = 0.0
    
    # Compute loss manually to save memory (Optimize)
    for c in range(C):
        cur_embeds = full_inputs_embeds[:, c]   # (B, L_T, D_llm)
        cur_mask = batch["target_attention_mask"][:, c] # (B, L_T)
        outputs = llm.model(
            inputs_embeds=cur_embeds,
            attention_mask=cur_mask,
            use_cache=False
        )
        hidden_states = outputs.last_hidden_state # (B, L_T, D_llm)
        
        chunk_loss_sum = 0.0
        for i in range(B):
            a_len = int(batch["a_len"][i].item())
            effective_len = int(cur_mask[i].sum().item())
            start_pos = effective_len - a_len
            end_pos = effective_len
            
            ans_hidden = hidden_states[i, start_pos-1:end_pos-1]   # (a_len, D_llm)
            ans_labels = batch["target_input_ids"][i, c, start_pos:end_pos]
            
            ans_logits = llm.lm_head(ans_hidden)
            chunk_loss = loss_fct(ans_logits, ans_labels)
            
            chunk_loss_sum += chunk_loss
        
        chunk_grad = torch.autograd.grad(
            outputs=chunk_loss_sum,
            inputs=cur_embeds,
            retain_graph=False,
            create_graph=False
        )[0]    # (B, L_T, D_llm)
        all_gradients[:, c] = chunk_grad
        total_loss_val += chunk_loss_sum
    
    return torch.tensor(total_loss_val, device=full_inputs_embeds.device), all_gradients


def compute_oracle_scores(loss_gradients: torch.Tensor, doc_repr: torch.Tensor, doclen_list: torch.Tensor, a_len: torch.Tensor, question_ids: torch.Tensor):
    """
    Compute the target scores for CA-Former based on the gradients of the oracle loss.
    Args:
        loss_gradients: Tensor of shape (B, C, L_T, D_llm)
        doc_repr: Tensor of shape (B*TOTAL_K, D_llm)
        doclen_list: Tensor of shape (B, TOTAL_K)
        a_len: Tensor of shape (B,)
        question_ids: Tensor of shape (B,)
    Returns:
        scores_oracle: List of length B. Each element is a Dict containing different
                       types of oracle scores for that specific batch item.
                       (All tensors are converted to numpy arrays for easier storage)
    """
    B, C, L_T, D = loss_gradients.shape
    doclen_list = doclen_list.view(B, C, -1)
    K = doclen_list.shape[2]
    TOTAL_K = C * K
    
    # Expand loss_gradients to match the shape of doc_repr
    # Convert FP32 -> BF16
    doc_repr = doc_repr.reshape(B, C, K, D)
    neg_loss_gradients = -1 * loss_gradients
    neg_loss_gradients = neg_loss_gradients.to(doc_repr.dtype)
    
    modes = ["whole", "marginal", "marginal-qa", "topk", "topk-qa"]
    # ========== Whole score computation ==========
    # Most naive way in parallel, faster
    # To reduce memory, we can average the gradients over the sequence length first
    avg_gradients = neg_loss_gradients.mean(dim=2)
    scores_oracle_whole = einops.einsum(doc_repr, avg_gradients,
                                "b c k d, b c d -> b c k")       # (B, C, K)
    
    # ========== Marginal & Top-k score computation ==========
    # More fine-grained way using sequential processing, slower
    TOPK_RANGE = 10
    scores_oracle_marginal = torch.zeros(B, C, K, device=loss_gradients.device)
    scores_oracle_marginal_qa = torch.zeros(B, C, K, device=loss_gradients.device)
    scores_oracle_topk = torch.zeros(B, C, K, TOPK_RANGE, device=loss_gradients.device)
    scores_oracle_topk_qa = torch.zeros(B, C, K, TOPK_RANGE, device=loss_gradients.device)
    
    for i in range(B):
        ans_length = int(a_len[i].item() if isinstance(a_len[i], torch.Tensor) else a_len[i])
        qa_len = ans_length + question_ids[i].shape[0]
        
        for c in range(C):
            start_pos = 0
            chunk_doc_len = int(doclen_list[i, c].sum().item())
            qa_gradients = neg_loss_gradients[i, c, chunk_doc_len : chunk_doc_len + qa_len]
            
            for k in range(K):
                end_pos = start_pos + int(doclen_list[i, c, k].item())
                # ----- Marginal -----
                marginal_gradients = neg_loss_gradients[i, c, start_pos:end_pos]   # (doc_len, D_llm)
                each_doc_repr = doc_repr[i, c, k]     # (D_llm,)
                each_scores_oracle = einops.einsum(each_doc_repr, marginal_gradients,
                                                    "d, l d -> l").mean(dim=-1)
                scores_oracle_marginal[i, c, k] = each_scores_oracle
                
                # ----- Marginal-QA -----
                marginal_qa_gradients = torch.cat([marginal_gradients, qa_gradients], dim=0)
                each_scores_oracle = einops.einsum(each_doc_repr, marginal_qa_gradients,
                                                "d, l d -> l").mean(dim=-1)
                scores_oracle_marginal_qa[i, c, k] = each_scores_oracle

                # ----- Top-k & Top-k-QA -----
                # norms = marginal_gradients.norm(p=2, dim=-1)
                all_token_scores = einops.einsum(each_doc_repr, marginal_gradients, "d, l d -> l")
                qa_token_scores = einops.einsum(each_doc_repr, qa_gradients, "d, l d -> l")

                actual_topk = min(TOPK_RANGE, marginal_gradients.shape[0])
                if actual_topk > 0:
                    topk_scores = torch.topk(all_token_scores, k=actual_topk).values
                    
                    all_qa_scores = torch.cat([all_token_scores, qa_token_scores], dim=0)
                    actual_topk_qa = min(TOPK_RANGE, all_qa_scores.shape[0])
                    topk_qa_scores = torch.topk(all_qa_scores, k=actual_topk_qa).values
                    
                    # Normalize
                    range_vectors = torch.arange(1, actual_topk + 1, device=topk_scores.device)
                    topk_cum_grads = torch.cumsum(topk_scores, dim=0) / range_vectors
                    range_vectors_qa = torch.arange(1, actual_topk_qa + 1, device=topk_qa_scores.device)
                    topk_qa_cum_grads = torch.cumsum(topk_qa_scores, dim=0) / range_vectors_qa
                    scores_oracle_topk[i, c, k, :actual_topk] = topk_cum_grads[:actual_topk]
                    scores_oracle_topk_qa[i, c, k, :actual_topk] = topk_qa_cum_grads[:actual_topk]
                start_pos = end_pos
    
    # ========== Flatten to List of Dicts ==========
    scores_oracle_whole = scores_oracle_whole.reshape(B, TOTAL_K)
    scores_oracle_marginal = scores_oracle_marginal.reshape(B, TOTAL_K)
    scores_oracle_marginal_qa = scores_oracle_marginal_qa.reshape(B, TOTAL_K)
    scores_oracle_topk = scores_oracle_topk.reshape(B, TOTAL_K, TOPK_RANGE)
    scores_oracle_topk_qa = scores_oracle_topk_qa.reshape(B, TOTAL_K, TOPK_RANGE)
        
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

    for r in range(world_size):
        search_pattern = f"{final_output_path}_rank{r}_part*.tmp"
        tmp_files = glob.glob(search_pattern)
        for tmp_path in tmp_files:
            if os.path.exists(tmp_path):
                try:
                    with h5py.File(tmp_path, "r") as h5file:
                        completed.update(set(h5file.keys()))
                except OSError as e:
                    print(f"[Warning] 파일 손상 감지됨: {tmp_path}. 해당 파일은 무시하고 진행.")
                    pass
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

def safe_open_h5(local_rank, filepath, mode="a"):
    try:
        return h5py.File(filepath, mode)
    except OSError as e:
        print(f"[Rank {local_rank}] 파일 손상 감지됨: {filepath}. 삭제 후 재생성합니다.")
        if os.path.exists(filepath):
            os.remove(filepath)
        return h5py.File(filepath, mode)


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
    # datamodule = GGDataModule(config)
    datamodule = GGChunkDataModule(config)
    # Prepare dataloaders
    if is_main_process:
        datamodule.prepare_data()
    if dist.is_initialized():
        dist.barrier()
    
    output_base_dir = os.path.join(config.output_dir, get_model_name(config.model.model_name))
    final_output_path = os.path.join(output_base_dir, config.output_file)

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
    
    final_output_path = os.path.join(output_base_dir, config.output_file)
    
    SAVE_SIZE = 100
    local_pattern = f"{final_output_path}_rank{local_rank}_part*.tmp"
    local_files = glob.glob(local_pattern)

    if not local_files:
        chunk_idx = 0
        current_samples = 0
    else:
        part_nums = [int(f.split("_part")[-1].split(".tmp")[0]) for f in local_files]
        chunk_idx = max(part_nums)
        last_file_path = f"{final_output_path}_rank{local_rank}_part{chunk_idx}.tmp"

        try:
            with h5py.File(last_file_path, "r") as f:
                current_samples = len(f.keys())
        except Exception:
            current_samples = 0
        
        if current_samples >= SAVE_SIZE:
            chunk_idx += 1
            current_samples = 0

    print(f"[Rank {local_rank}] Starting processing with chunk index {chunk_idx} and {current_samples} samples.")
    
    def get_tmp_path(rank, c_idx):
        return f"{final_output_path}_rank{rank}_part{c_idx}.tmp"
    temp_output_path = get_tmp_path(local_rank, chunk_idx)
    h5file = safe_open_h5(temp_output_path, "a")


    for _split, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:
        logger.info(f"Processing {_split} split...")
        iterator = tqdm(dataloader, desc=f"[Rank {local_rank}] Processing {_split}", position=local_rank)

        for batch in iterator:
            batch_ids = [str(idx.item()) for idx in batch["idx"]]
            # Skip already processed batches
            if all(b_id in global_completed_ids for b_id in batch_ids):
                continue

            batch = {k: v.to(local_rank) if hasattr(v, "to") else v for k, v in batch.items()}
            doc_repr = forward(llm, batch)

            with torch.enable_grad():
                loss_oracle, loss_gradients = compute_oracle_loss_gradients(llm, batch)
                scores_oracle = compute_oracle_scores(loss_gradients, doc_repr, batch["doclen_list"], batch["a_len"], batch["question_ids"])
            
            # Store the gradients in the precompute table
            for i in range(batch["idx"].shape[0]):
                example_id = str(batch["idx"][i].item())

                if example_id not in h5file:
                    grp = h5file.create_group(example_id)
                    for score_type, score_values in scores_oracle[i].items():
                        grp.create_dataset(score_type, data=score_values)
                
                current_samples += 1
                if current_samples >= SAVE_SIZE:
                    h5file.close()
                    chunk_idx += 1
                    temp_output_path = get_tmp_path(local_rank, chunk_idx)
                    h5file = safe_open_h5(temp_output_path, "a")
                    current_samples = 0
    if h5file:
        h5file.close()

    if dist.is_initialized():
        dist.barrier()
    if is_main_process:
        merge_h5_files(final_output_path, world_size)
    cleanup_ddp()


if __name__ == "__main__":
    main()