from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf, ListConfig
from transformers import AutoModelForCausalLM, AutoModel, LlamaForCausalLM
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
import glob
import torch.nn.functional as F
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

    search_pattern = f"{output_path}_rank*_part*.tmp"
    temp_files = glob.glob(search_pattern)
    if not temp_files:
        logger.warning(f"No temporary files found matching: {search_pattern}")
        return
    temp_files.sort()

    duplicate_count = 0
    merged_count = 0

    with h5py.File(output_path, "w") as final_h5:
        for temp_path in tqdm(temp_files, desc="Merging files"):
            with h5py.File(temp_path, "r") as temp_h5:
                for group_name in temp_h5.keys():
                    if group_name in final_h5:
                        duplicate_count += 1
                    else:
                        temp_h5.copy(group_name, final_h5)
                        merged_count += 1
            
            # os.remove(temp_path)
    logger.info(f"Successfully merged {len(temp_files)} files into {output_path}")
    logger.info(f"Total entries merged: {merged_count}")
    if duplicate_count > 0:
        logger.warning(f"Skipped {duplicate_count} duplicate keys during merge.")


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
    # llm.enable_input_require_grads()
    # llm.gradient_checkpointing_enable()
    llm.eval()

def forward(llm: AutoModelForCausalLM, batch: Any):
    """
    Forward pass to generate LLM hidden states for both documents and CA-Former input.
    Returns:
        doc_repr: Tensor of shape (B*K, D_llm) - LLM representation of the document (using <EOS> token)
    """
    # Generate LLM hidden states of documents, last layer only, <EOS> token
    # Since we use right-padding, we should track the position of <EOS> token
    with torch.no_grad():
        doc_outputs = llm.model(
            input_ids=batch["doc_input_ids"],
            attention_mask=batch["doc_attention_mask"],
            use_cache=False
        )
        last_hidden_state = doc_outputs.last_hidden_state  # (B*K, L, D_llm)
        last_token_indices = batch["doc_attention_mask"].sum(dim=-1) - 1   # (B*K,)
        D_llm = last_hidden_state.shape[-1]
        gather_indices = last_token_indices.view(-1, 1, 1).expand(-1, 1, D_llm)
        doc_repr = torch.gather(last_hidden_state, dim=1, index=gather_indices).squeeze(1)  # (B*K, D_llm)
    return doc_repr

# TODO: Change the logic -> Fixed!
def compute_oracle_loss_gradients(llm: AutoModelForCausalLM, doc_repr: torch.Tensor, batch):
    """
    Compute the gradients conditioned on the mixed_doc_repr of the oracle loss.
    Basically, we return the whole gradient matrix.
    Args:
        doc_repr: Tensor of shape (B*K, D_llm)
    Returns:
        loss_oracle: scalar tensor
        loss_gradients: Tensor of shape (B, D_llm)
    """
    B = len(batch["idx"])
    K = doc_repr.shape[0] // B

    # Make inputs_embeds for the oracle loss calculation
    score_input_ids = batch["score_input_ids"]  # (B, 1+q_len+a_len)
    score_attention_mask = batch["score_attention_mask"]  # (B, 1+q_len+a_len)

    batched_doc_repr = doc_repr.view(B, K, -1)
    mixed_doc_repr = torch.mean(batched_doc_repr, dim=1)  # (B, D_llm)
    mixed_doc_repr = mixed_doc_repr.detach().requires_grad_(True)

    # Use torch.cat instead of indexing (in-place error)
    query_ans_repr = llm.get_input_embeddings()(score_input_ids[:, 1:])
    score_repr = torch.cat([
        mixed_doc_repr.unsqueeze(1), query_ans_repr
    ], dim=1)

    # Vectorized label creation for oracle loss
    end_pos = score_attention_mask.sum(dim=-1)    # (B,)
    start_pos = end_pos - batch["a_len"]          # (B,)

    start_pos = start_pos.unsqueeze(1)  # (B, 1)
    end_pos = end_pos.unsqueeze(1)      # (B, 1)
    positions = torch.arange(score_input_ids.shape[1], device=score_input_ids.device)
    mask = (positions >= start_pos) & (positions < end_pos)
    labels = torch.full_like(score_input_ids, fill_value=-100)
    labels[mask] = score_input_ids[mask]

    # Compute loss and gradients manually to save memory
    outputs = llm.model(
        inputs_embeds=score_repr,
        attention_mask=score_attention_mask,
        use_cache=False
    )
    hidden_states = outputs.last_hidden_state

    loss_fct = torch.nn.CrossEntropyLoss()
    shift_hidden_states = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_losses = []
    for i in range(B):
        curr_start = start_pos[i] - 1
        valid_h = shift_hidden_states[i, curr_start:]
        valid_l = shift_labels[i, curr_start:]

        valid_logits = llm.lm_head(valid_h)

        loss_i = loss_fct(valid_logits, valid_l)
        batch_losses.append(loss_i)
    loss_oracle = torch.stack(batch_losses).mean()

    # Compute gradients w.r.t. mixed_doc_repr
    loss_gradients = torch.autograd.grad(
        loss_oracle,
        mixed_doc_repr,
        retain_graph=False,
        create_graph=False,
    )[0]

    return loss_oracle, loss_gradients

# TODO: Change the logic
def compute_oracle_scores(loss_gradients: torch.Tensor, doc_repr: torch.Tensor):
    """
    Compute the target scores for CA-Former based on the gradients of the oracle loss.
    Args:
        loss_gradients: Tensor of shape (B, D_llm)
        doc_repr: Tensor of shape (B*k, D_llm)
    Returns:
        scores_oracle: List of length B. Each element is a Dict containing different
                       types of oracle scores for that specific batch item.
                       (All tensors are converted to numpy arrays for easier storage)
    """
    B, D = loss_gradients.shape
    K = doc_repr.shape[0] // loss_gradients.shape[0]
    # Expand loss_gradients to match the shape of doc_repr
    batched_doc_repr = doc_repr.view(B, K, D)  # (B, K, D)
    neg_grad = -loss_gradients.unsqueeze(1)

    # scores_oracle = einops.einsum(doc_repr, neg_loss_gradients_expanded,
    #                               "b d, b d -> b").reshape(B, -1)   # (B, K)
    scores_oracle = F.cosine_similarity(batched_doc_repr, neg_grad, dim=-1)
    formatted_scores = []
    modes = ["oracle"]
    # ========== Flatten to List of Dicts ==========
    for i in range(B):
        item_scores = {
            "oracle": scores_oracle[i].detach().cpu().float().numpy(),
        }
        formatted_scores.append(item_scores)
    
    return formatted_scores


##### Function Type Two #####
def compute_oracle_loss_gradients_isolated(llm: AutoModelForCausalLM, doc_repr: torch.Tensor, batch):
    B = len(batch["idx"])
    K = doc_repr.shape[0] // B

    # Make inputs_embeds for the oracle loss calculation
    score_input_ids = batch["score_input_ids"]  # (B, K+q_len+a_len)
    score_attention_mask = batch["score_attention_mask"]  # (B, K+q_len+a_len), used for only vectorized label creation

    batched_doc_repr = doc_repr.view(B, K, -1).requires_grad_(True)
    # Use torch.cat instead of indexing (in-place error)
    query_ans_repr = llm.get_input_embeddings()(score_input_ids[:, K:])
    score_repr = torch.cat([
        batched_doc_repr, query_ans_repr
    ], dim=1)
    
    # TODO: Make 4D attention mask & position ids
    L = score_repr.shape[1]
    causal_mask = torch.tril(torch.ones((L, L), device=score_repr.device, dtype=torch.bool))
    causal_mask[:K, :K] = torch.diag(torch.ones(K, device=score_repr.device, dtype=torch.bool))
    attention_mask_4d = torch.zeros((B, 1, L, L), device=score_repr.device, dtype=torch.bfloat16)
    attention_mask_4d = attention_mask_4d.masked_fill(~causal_mask, float("-inf"))
    padding_mask = (score_attention_mask == 0).view(B, 1, 1, L)
    attention_mask_4d = attention_mask_4d.masked_fill(padding_mask, float("-inf"))

    parallel_position_ids = torch.zeros(K, device=score_repr.device, dtype=torch.long)
    qa_position_ids = torch.arange(1, L - K + 1, device=score_repr.device, dtype=torch.long)
    position_ids = torch.cat([
        parallel_position_ids,
        qa_position_ids
    ], dim=0).unsqueeze(0)  # (1, L)

    # Vectorized label creation for oracle loss
    end_pos = score_attention_mask.sum(dim=-1)    # (B,)
    start_pos = end_pos - batch["a_len"]          # (B,)

    start_pos = start_pos.unsqueeze(1)  # (B, 1)
    end_pos = end_pos.unsqueeze(1)      # (B, 1)
    positions = torch.arange(score_input_ids.shape[1], device=score_input_ids.device)
    mask = (positions >= start_pos) & (positions < end_pos)
    labels = torch.full_like(score_input_ids, fill_value=-100)
    labels[mask] = score_input_ids[mask]

    # Compute loss and gradients manually to save memory
    outputs = llm.model(
        inputs_embeds=score_repr,
        attention_mask=attention_mask_4d,
        use_cache=False,
        position_ids=position_ids
    )
    hidden_states = outputs.last_hidden_state

    loss_fct = torch.nn.CrossEntropyLoss()
    shift_hidden_states = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_losses = []
    for i in range(B):
        curr_start = start_pos[i] - 1
        valid_h = shift_hidden_states[i, curr_start:]
        valid_l = shift_labels[i, curr_start:]

        valid_logits = llm.lm_head(valid_h)

        loss_i = loss_fct(valid_logits, valid_l)
        batch_losses.append(loss_i)
    loss_oracle = torch.stack(batch_losses).sum()

    # Compute gradients w.r.t. batched_doc_repr
    loss_gradients = torch.autograd.grad(
        loss_oracle,
        batched_doc_repr,
        retain_graph=False,
        create_graph=False,
    )[0]

    return loss_oracle, loss_gradients


def compute_oracle_scores_isolated(loss_gradients: torch.Tensor, doc_repr: torch.Tensor):
    """
    Args:
        loss_gradients: Tensor of shape (B, K, D_llm)
        doc_repr: Tensor of shape (B*K, D_llm)
    """
    B, K, D = loss_gradients.shape
    # Expand loss_gradients to match the shape of doc_repr
    batched_doc_repr = doc_repr.view(B, K, D)  # (B, K, D)
    neg_grad = -loss_gradients

    scores_oracle = torch.sum(batched_doc_repr * neg_grad, dim=-1)  # (B, K)
    # scores_oracle = F.cosine_similarity(batched_doc_repr, neg_grad, dim=-1)
    formatted_scores = []
    modes = ["oracle"]
    # ========== Flatten to List of Dicts ==========
    for i in range(B):
        item_scores = {
            "oracle": scores_oracle[i].detach().cpu().float().numpy(),
        }
        formatted_scores.append(item_scores)
    
    return formatted_scores
##### Function Type Two #####


##### Function Type Three (Leave-One-Out) #####
def compute_oracle_scores_loo(llm: AutoModelForCausalLM, doc_repr: torch.Tensor, batch):
    pass



##### Function Type Three (Leave-One-Out) #####



##### Function Type Four #####
def compute_oracle_loss_gradients_multi(llm: AutoModelForCausalLM, doc_repr: torch.Tensor, batch):
    B = batch["idx"].shape[0]
    K = doc_repr.shape[0] // B

    # Make inputs_embeds for the oracle loss calculation
    score_input_ids = batch["score_input_ids"]  # (B, K+q_len+a_len)
    score_attention_mask = batch["score_attention_mask"]  # (B, K+q_len+a_len), used for only vectorized label creation

    batched_doc_repr = doc_repr.view(B, K, -1).requires_grad_(True)
    # Use torch.cat instead of indexing (in-place error)
    query_ans_repr = llm.get_input_embeddings()(score_input_ids[:, K:])
    score_repr = torch.cat([
        batched_doc_repr, query_ans_repr
    ], dim=1)
    
    # TODO: Make 4D attention mask & position ids
    L = score_repr.shape[1]
    causal_mask = torch.tril(torch.ones((L, L), device=score_repr.device, dtype=torch.bool))
    causal_mask[:K, :K] = torch.diag(torch.ones(K, device=score_repr.device, dtype=torch.bool))
    attention_mask_4d = torch.zeros((B, 1, L, L), device=score_repr.device, dtype=torch.bfloat16)
    attention_mask_4d = attention_mask_4d.masked_fill(~causal_mask, float("-inf"))
    padding_mask = (score_attention_mask == 0).view(B, 1, 1, L)
    attention_mask_4d = attention_mask_4d.masked_fill(padding_mask, float("-inf"))

    parallel_position_ids = torch.zeros(K, device=score_repr.device, dtype=torch.long)
    qa_position_ids = torch.arange(1, L - K + 1, device=score_repr.device, dtype=torch.long)
    position_ids = torch.cat([
        parallel_position_ids,
        qa_position_ids
    ], dim=0).unsqueeze(0)  # (1, L)

    # Vectorized label creation for oracle loss
    end_pos = score_attention_mask.sum(dim=-1)    # (B,)
    start_pos = end_pos - batch["a_len"]          # (B,)

    start_pos = start_pos.unsqueeze(1)  # (B, 1)
    end_pos = end_pos.unsqueeze(1)      # (B, 1)
    positions = torch.arange(score_input_ids.shape[1], device=score_input_ids.device)
    mask = (positions >= start_pos) & (positions < end_pos)
    labels = torch.full_like(score_input_ids, fill_value=-100)
    labels[mask] = score_input_ids[mask]

    # Compute loss and gradients manually to save memory
    outputs = llm.model(
        inputs_embeds=score_repr,
        attention_mask=attention_mask_4d,
        use_cache=False,
        position_ids=position_ids
    )
    hidden_states = outputs.last_hidden_state

    loss_fct = torch.nn.CrossEntropyLoss()
    shift_hidden_states = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_losses = []
    for i in range(B):
        curr_start = start_pos[i] - 1
        valid_h = shift_hidden_states[i, curr_start:]
        valid_l = shift_labels[i, curr_start:]

        valid_logits = llm.lm_head(valid_h)

        loss_i = loss_fct(valid_logits, valid_l)
        batch_losses.append(loss_i)
    loss_oracle = torch.stack(batch_losses).sum()

    # Compute gradients w.r.t. batched_doc_repr
    loss_gradients = torch.autograd.grad(
        loss_oracle,
        batched_doc_repr,
        retain_graph=False,
        create_graph=False,
    )[0]

    return loss_oracle, loss_gradients


def compute_oracle_scores_multi(loss_gradients: torch.Tensor, doc_repr: torch.Tensor):
    """
    Args:
        loss_gradients: Tensor of shape (B, K, D_llm)
        doc_repr: Tensor of shape (B*K, D_llm)
    """
    B, K, D = loss_gradients.shape
    # Expand loss_gradients to match the shape of doc_repr
    batched_doc_repr = doc_repr.view(B, K, D)  # (B, K, D)
    neg_grad = -loss_gradients

    scores_oracle = torch.sum(batched_doc_repr * neg_grad, dim=-1)  # (B, K)
    # scores_oracle = F.cosine_similarity(batched_doc_repr, neg_grad, dim=-1)
    formatted_scores = []
    modes = ["oracle"]
    # ========== Flatten to List of Dicts ==========
    for i in range(B):
        item_scores = {
            "oracle": scores_oracle[i].detach().cpu().float().numpy(),
        }
        formatted_scores.append(item_scores)
    
    return formatted_scores
##### Function Type Four #####



def get_completed_ids(world_size, final_output_path):
    completed = set()
    # import pdb; pdb.set_trace()
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


### Two types of oracle loss and score functions for experimentation
ORACLE_LOSS_GRADIENTS_FUNC = {
    "base": compute_oracle_loss_gradients,
    "isolated": compute_oracle_loss_gradients_isolated,
    "loo": None,
    "multi": None
}
ORACLE_SCORE_FUNC = {
    "base": compute_oracle_scores,
    "isolated": compute_oracle_scores_isolated,
    "loo": compute_oracle_scores_loo,
    "multi": None
}


def main():
    torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf, omegaconf.base.ContainerMetadata])
    # Allow TF32 (This can be useful for mixed precision training)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    local_rank, world_size = setup_ddp()
    is_main_process = (local_rank == 0)

    config = load_config()
    seed_everything(config.seed)
    experiment_name = "gen_emb_precompute_table"
    config.output_dir = os.path.join(config.output_dir, experiment_name)

    if is_main_process:
        setup_logger("main", config.output_dir)
        logger = logging.getLogger(__name__)
        logger.info("Configuration Loaded:")
        logger.info(OmegaConf.to_yaml(config))
    else:
        logger = logging.getLogger(__name__)

    # Load datamodule & model
    datamodule = GGEmbDataModule(config)
    # Prepare dataloaders
    if is_main_process:
        datamodule.prepare_data()
    if dist.is_initialized():
        dist.barrier()
    
    output_base_dir = os.path.join(config.output_dir, get_model_name(config.model.model_name))
    # final_output_path = os.path.join(output_base_dir, f"hotpotqa-d_val_emb_precompute_table_{config.data.topk_per_query}_{config.oracle_mode}.h5")
    final_output_path = config.data.precompute_table_path

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

    llm, _ = load_model(config.model.model_name, custom_attn=(config.oracle_mode == "isolated"))
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
    h5file = safe_open_h5(local_rank, temp_output_path, "a")

    oracle_loss_func = ORACLE_LOSS_GRADIENTS_FUNC[config.oracle_mode]
    oracle_score_func = ORACLE_SCORE_FUNC[config.oracle_mode]

    for _split, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:
        logger.info(f"Processing {_split} split...")
        iterator = tqdm(dataloader, desc=f"[Rank {local_rank}] Processing {_split}", position=local_rank)

        for batch in iterator:
            batch_ids = [str(idx) for idx in batch["idx"]]
            # Skip already processed batches
            if all(b_id in global_completed_ids for b_id in batch_ids):
                continue

            batch = {k: v.to(local_rank) if hasattr(v, "to") else v for k, v in batch.items()}
            doc_repr = forward(llm, batch)

            with torch.enable_grad():
                if config.oracle_mode != "loo":
                    loss_oracle, loss_gradients = oracle_loss_func(llm, doc_repr, batch)
                    scores_oracle = oracle_score_func(loss_gradients, doc_repr)
                else:
                    scores_oracle = oracle_score_func(llm, doc_repr, batch)

            # Store the gradients in the precompute table
            for i in range(len(batch["idx"])):
                example_id = str(batch["idx"][i])

                if example_id in global_completed_ids:
                    continue

                if example_id not in h5file:
                    grp = h5file.create_group(example_id)
                    for score_type, score_values in scores_oracle[i].items():
                        grp.create_dataset(score_type, data=score_values)
                    global_completed_ids.add(example_id)

                current_samples += 1
                if current_samples >= SAVE_SIZE:
                    h5file.close()
                    chunk_idx += 1
                    temp_output_path = get_tmp_path(local_rank, chunk_idx)
                    h5file = safe_open_h5(local_rank, temp_output_path, "a")
                    current_samples = 0
    if h5file:
        h5file.close()
    if dist.is_initialized():
        dist.barrier()

    if is_main_process:
        merge_h5_files(final_output_path, world_size)
    if dist.is_initialized():
        dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    main()