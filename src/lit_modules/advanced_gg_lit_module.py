import torch.nn as nn
import torch
import logging
import wandb
import einops
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Union
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM
from torch.nn.attention.flex_attention import create_block_mask

from utils import compute_metrics, parse_reference_answer
from models import load_model, CAFormerGGClassifier

logger = logging.getLogger(__name__)


class AdvancedGGLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig,
                 llm: AutoModelForCausalLM, llm_tokenizer: AutoTokenizer, caformer_clf: CAFormerGGClassifier):
        super().__init__()
        self.save_hyperparameters(ignore=["llm", "caformer_clf"])

        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.caformer_clf = caformer_clf
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.n_iter = getattr(cfg, "n_iter", 1)
        self.oracle_mode = getattr(cfg, "oracle_mode", "whole")
        self.lmbda = getattr(cfg, "lmbda", 1.0) 

        self.score_transform = getattr(cfg, "score_transform", None)
        self.scaling_factor = getattr(cfg, "scaling_factor", 1.0)
        self.T = getattr(cfg, "T", 1.0)
        # self.guide_loss_fn = nn.L1Loss() if self.score_transform is None else nn.BCEWithLogitsLoss()
        self.guide_loss_fn = nn.MSELoss()
        self.gen_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.val_logits = []
        self.val_labels = []
        self.compiled_create_block_mask = torch.compile(create_block_mask)
        self.prepare_modules()

    def prepare_modules(self):
        # Freeze the pretrained LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        
        for param in self.caformer_clf.parameters():
            param.requires_grad = True
            
        # Gradient checkpointing configuration
        # This is useful for fine-tuning adapter weights while keeping the model weights fixed.
        self.llm.enable_input_require_grads()
        self.llm.gradient_checkpointing_enable()
        if hasattr(self.llm.config, "attention_dropout"):
            self.llm.config.attention_dropout = 0.0
        if hasattr(self.llm.config, "dropout"):
            self.llm.config.dropout = 0.0

        # torch.compile(self.llm.model) is disabled: compiled submodule inside a DDP-wrapped
        # outer module can cause CUDA device-side asserts due to stream sync issues.
        # flex_attention already compiles the attention kernel internally via compile_friendly_flex_attention.
        # if getattr(self.cfg, "do_isolate", False):
        #     self.llm.model = torch.compile(self.llm.model, dynamic=True)

    def train(self, mode: bool = True):
        super().train(mode)
        # self.llm.eval()  # Ensure the base model is always in eval mode

    def convert_document_ids_to_tensor(self, padded_inputs_embeds, doclen_list, query_hidden_states, question_attention_mask):
        """
        Convert document lengths and question ids to a tensor of shape (B, L_total)
        Outputs will be like [0, 0, 1, 1, 1, ..., k, k, -1, -1, ..., -100, -100]
        Args:
            doclen_list: Tensor of shape (B, k)
            query_hidden_states: Tensor of shape (B*k, S, D_llm)
            question_attention_mask: Tensor of shape (B, q_len)
        Returns:
            document_ids_tensor: Tensor of shape (B, L_total)
        """
        B, K = doclen_list.shape
        tgt_len = padded_inputs_embeds.shape[1]
        query_vec_len = query_hidden_states.shape[1]
        device = padded_inputs_embeds.device

        document_ids_tensor = torch.full((B, tgt_len), -100, dtype=torch.long, device=device)
        pos = torch.arange(tgt_len, device=device).unsqueeze(0).expand(B, tgt_len)

        block_lens = doclen_list + query_vec_len
        end_offsets = torch.cumsum(block_lens, dim=1)  # (B, K)
        start_offsets = end_offsets - block_lens  # (B, K)

        for j in range(K):
            start = start_offsets[:, j:j+1] # (B, 1)
            end = end_offsets[:, j:j+1]   # (B, 1)
            mask = (pos >= start) & (pos < end)
            document_ids_tensor = torch.where(mask, j, document_ids_tensor)

        q_start = end_offsets[:, -1:]  # (B, 1)
        valid_q_len = question_attention_mask.sum(dim=-1, keepdim=True)
        q_mask = (pos >= q_start) & (pos < q_start + valid_q_len)
        document_ids_tensor = torch.where(q_mask, -1, document_ids_tensor)

        return document_ids_tensor

    def forward(self, batch):
        """
        Forward pass to generate LLM hidden states for both documents and CA-Former input.
        Returns:
            doc_repr: Tensor of shape (B*K, D_llm) - LLM representation of the document (using <EOS> token)
            llm_repr: Tensor of shape (B*K, L, S, D_llm) - LLM hidden states for CA-Former input
        """
        # Generate LLM hidden states for CA-Former input
        with torch.no_grad():
            llm_outputs = self.llm(
                input_ids=batch["source_input_ids"],
                attention_mask=batch["source_attention_mask"],
                output_hidden_states=True,
            )
            llm_repr = torch.stack(llm_outputs.hidden_states[-12:]).permute(1, 0, 2, 3)   # (B*K, L, S, D_llm)
        return llm_repr
    
    def get_doc_repr(self, batch):
        # Generate LLM hidden states of documents, last layer only, <EOS> token
        # Since we use right-padding, we should track the position of <EOS> token
        with torch.no_grad():
            doc_outputs = self.llm(
                input_ids=batch["doc_input_ids"],
                attention_mask=batch["doc_attention_mask"],
                output_hidden_states=True,
            )
            last_hidden_state = doc_outputs.hidden_states[-1]  # (B*K, L, D_llm)
            last_token_indices = batch["doc_attention_mask"].sum(dim=-1) - 1   # (B*K,)
            D_llm = last_hidden_state.shape[-1]
            gather_indices = last_token_indices.view(-1, 1, 1).expand(-1, 1, D_llm)
            doc_repr = torch.gather(last_hidden_state, dim=1, index=gather_indices).squeeze(1)  # (B*K, D_llm)
        return doc_repr
    
    def compute_oracle_loss_gradients(self, batch):
        """
        Compute the gradients of the oracle loss.
        Basically, we return the whole gradient matrix.
        Returns:
            loss_oracle: scalar tensor
            loss_gradients: Tensor of shape (B, k * max_seq_length + max_ans_length, D_llm)
        """
        inputs_embeds = self.llm.get_input_embeddings()(batch["target_input_ids"])
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        llm_outputs = self.llm(
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
        return loss_oracle.detach(), loss_gradients

    def compute_oracle_scores(self, loss_gradients, doc_repr, doclen_list):
        """
        Compute the target scores for CA-Former based on the gradients of the oracle loss.
        Args:
            loss_gradients: Tensor of shape (B, k * max_seq_length + max_ans_length, D_llm)
            doc_repr: Tensor of shape (B*k, D_llm)
            doclen_list: Tensor of shape (B, k)
        Returns:
            scores_oracle: Tensor of shape (B*k, 1)
        """
        B = loss_gradients.shape[0]
        K = doc_repr.shape[0] // loss_gradients.shape[0]
        D = doc_repr.shape[-1]
        # Expand loss_gradients to match the shape of doc_repr
        loss_gradients_expanded = loss_gradients.unsqueeze(1).expand(-1, K, -1, -1)
        loss_gradients_expanded = loss_gradients_expanded.reshape(B*K, -1, D)
        neg_loss_gradients_expanded = -1 * loss_gradients_expanded

        # Most naive way in parallel, faster
        if self.oracle_mode == "whole":
            scores_oracle = einops.einsum(doc_repr, neg_loss_gradients_expanded,
                                      "b d, b l d -> b l").mean(dim=-1)
        # More fine-grained way using sequential processing, slower
        elif self.oracle_mode == "marginal":
            scores_oracle = torch.zeros(B*K, device=loss_gradients.device)
            for i in range(B):
                start_pos = 0
                for j in range(K):
                    flat_idx = i*K + j
                    end_pos = start_pos + doclen_list[i, j]
                    marginal_gradients = neg_loss_gradients_expanded[flat_idx, start_pos:end_pos]   # (doc_len, D_llm)
                    each_doc_repr = doc_repr[flat_idx]     # (D_llm,)
                    each_scores_oracle = einops.einsum(each_doc_repr, marginal_gradients,
                                                       "d, l d -> l").mean(dim=-1)
                    scores_oracle[flat_idx] = each_scores_oracle
                    start_pos = end_pos
        else:
            raise ValueError(f"Unknown oracle mode: {self.oracle_mode}")
        # scaling up
        scores_oracle = scores_oracle * self.scaling_factor
        return scores_oracle
    
    def compute_interleaving_loss(self, batch, query_hidden_states, return_logits=False):
        """
        Compute the generation loss conditioned on the interleaving document inputs.
        interleaving document inputs: D_1 + Q_{CA}^{(1)} + ... + D_K + Q_{CA}^{(K)} + Q + A (Teacher-forcing)
        Args:
            batch: dict containing the following keys:
                target_input_ids: Tensor of shape (B, L_total)
                target_attention_mask: Tensor of shape (B, L_total)
                doclen_list: Tensor of shape (B, k)
                a_len: Tensor of shape (B,)
            query_hidden_states: Tensor of shape (B*k, S, D_llm)
        Returns:
            gen_loss: scalar tensor
            (Optional) batch_logits: length of B List of Tensor of shape (A, V)
            (Optional) batch_labels: length of B List of Tensor of shape (A,)
        """
        target_input_ids = batch["target_input_ids"]
        target_attention_mask = batch["target_attention_mask"]
        doclen_list = batch["doclen_list"]
        a_len_list = batch["a_len"]
        question_attention_mask = batch["question_attention_mask"]

        full_repr = self.llm.get_input_embeddings()(target_input_ids)   # (B, L_total, D_llm)
        B, K = doclen_list.shape
        S = query_hidden_states.shape[1]
        D = full_repr.shape[-1]
        max_len = self.cfg.max_interleaving_len
        device = full_repr.device

        padded_inputs_embeds = torch.zeros((B, max_len, D), device=device, dtype=full_repr.dtype)
        padded_target_labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)
        padded_attention_mask = torch.zeros((B, max_len), device=device, dtype=torch.long)
        
        # Vectorized way
        block_lens = doclen_list + S
        cum_block_lens = torch.cumsum(block_lens, dim=1)
        doc_starts = cum_block_lens - block_lens
        query_starts = doc_starts + doclen_list

        for i in range(B):
            orig_ptr = 0
            sample_doclen_sum = doclen_list[i].sum()
            e_len = target_attention_mask[i].sum()
            a_len = a_len_list[i]
            
            for j in range(K):
                d_len = doclen_list[i, j]
                d_start = doc_starts[i, j]
                q_start = query_starts[i, j]
                
                padded_inputs_embeds[i, d_start:d_start + d_len] = full_repr[i, orig_ptr:orig_ptr + d_len]
                padded_inputs_embeds[i, q_start:q_start + S] = query_hidden_states[i*K + j]

                orig_ptr += d_len
                
            qa_start_interleaved = cum_block_lens[i, -1]
            remaining_len = (e_len - sample_doclen_sum)
            valid_qa_len = torch.min(remaining_len, max_len - qa_start_interleaved).clamp(min=0)

            padded_inputs_embeds[i, qa_start_interleaved:qa_start_interleaved + valid_qa_len] = \
                full_repr[i, sample_doclen_sum:sample_doclen_sum + valid_qa_len]
            
            total_active_len = qa_start_interleaved + valid_qa_len
            padded_attention_mask[i, :total_active_len] = 1
            
            label_start = (total_active_len - a_len).clamp(min=0)
            padded_target_labels[i, label_start : total_active_len] = target_input_ids[i, e_len - a_len : e_len]

        # Generate logits and compute loss
        # GPU MEMORY BOTTLENECK!! (~13GB for 10 docs)
        # Document-Isolated Masking
        if getattr(self.cfg, "do_isolate", False):
            document_ids = self.convert_document_ids_to_tensor(
                padded_inputs_embeds, doclen_list, query_hidden_states, question_attention_mask)
            document_ids = document_ids.to(torch.int32)
            # document_ids: (B, T), values: j in [0,K-1] (doc+Q_CA blocks), -1 (query), -100 (padding)

            def document_mask_mod(b, h, q_idx, kv_idx) -> bool:
                causal_mask = q_idx >= kv_idx

                q_id = document_ids[b, q_idx]
                kv_id = document_ids[b, kv_idx]

                is_identical_doc = q_id == kv_id
                is_question = q_id == -1

                is_valid = (q_id != -100) & (kv_id != -100)
                return causal_mask & is_valid & (is_identical_doc | is_question)
            
            B, T, _ = padded_inputs_embeds.shape
            block_masks = self.compiled_create_block_mask(
                document_mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, BLOCK_SIZE=128, device=device)
            padded_attention_mask = block_masks

            # T_seq = document_ids.shape[1]
            # pos = torch.arange(T_seq, device=device)
            # # causal_mask[q, kv] = (q >= kv)
            # causal_mask = (pos.unsqueeze(1) >= pos.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

            # q_doc = document_ids.unsqueeze(2)   # (B, T, 1)
            # kv_doc = document_ids.unsqueeze(1)  # (B, 1, T)
            # is_same_doc = (q_doc == kv_doc)                          # (B, T, T)
            # is_question = (q_doc == -1)                              # (B, T, T)
            # is_valid = (q_doc != -100) & (kv_doc != -100)           # (B, T, T)
            # doc_mask = (is_valid & (is_same_doc | is_question)).unsqueeze(1)  # (B, 1, T, T)

            # bool_mask = (causal_mask & doc_mask)         # (B, 1, T, T) bool
            # dtype = full_repr.dtype
            # min_val = torch.finfo(dtype).min

            # padded_attention_mask = torch.zeros_like(bool_mask, dtype=dtype)
            # padded_attention_mask.masked_fill_(~bool_mask, min_val)

        outputs = self.llm.model(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=padded_attention_mask,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state  # (B, L_total, D_llm)

        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = padded_target_labels[:, 1:].contiguous()

        mask = shift_labels != -100
        valid_hidden = shift_hidden[mask]   # (a_len, D)
        valid_labels = shift_labels[mask]   # (a_len,)

        ans_logits = self.llm.lm_head(valid_hidden)   # (a_len, V)
        gen_loss = self.gen_loss_fn(ans_logits, valid_labels)

        batch_logits, batch_labels = [], []
        if return_logits:
            all_preds = ans_logits.argmax(dim=-1).detach().cpu()
            all_labels = valid_labels.detach().cpu()
            curr = 0
            for i in range(B):
                length = int(a_len_list[i].item())
                batch_logits.append(all_preds[curr:curr + length])
                batch_labels.append(all_labels[curr:curr + length])
                curr += length

        outputs = (gen_loss,)
        if return_logits:
            outputs += (batch_logits, batch_labels)
        
        return outputs
    
    # def compute_interleaving_loss(self, batch, query_hidden_states, return_logits=False):
    #     """
    #     Compute the generation loss conditioned on the interleaving document inputs.
    #     interleaving document inputs: D_1 + Q_{CA}^{(1)} + ... + D_K + Q_{CA}^{(K)} + Q + A (Teacher-forcing)
    #     Args:
    #         batch: dict containing the following keys:
    #             target_input_ids: Tensor of shape (B, L_total)
    #             target_attention_mask: Tensor of shape (B, L_total)
    #             doclen_list: Tensor of shape (B, k)
    #             a_len: Tensor of shape (B,)
    #         query_hidden_states: Tensor of shape (B*k, S, D_llm)
    #     Returns:
    #         gen_loss: scalar tensor
    #         (Optional) batch_logits: length of B List of Tensor of shape (A, V)
    #         (Optional) batch_labels: length of B List of Tensor of shape (A,)
    #     """
    #     target_input_ids = batch["target_input_ids"]
    #     target_attention_mask = batch["target_attention_mask"]
    #     doclen_list = batch["doclen_list"]
    #     a_len_list = batch["a_len"]
    #     question_attention_mask = batch["question_attention_mask"]

    #     full_repr = self.llm.get_input_embeddings()(target_input_ids)   # (B, L_total, D_llm)
    #     B, K = doclen_list.shape
    #     _, _, D = full_repr.shape
        
    #     input_embeds_list = []
    #     attention_mask_list = []
    #     labels_list = []
        
    #     for i in range(B):
    #         start_pos = 0
    #         sample_input_embeds = []
            
    #         effective_len = int(target_attention_mask[i].sum().item())
    #         a_len = int(a_len_list[i].item())
            
    #         for j in range(K):
    #             flat_idx = i*K + j
    #             doclen = int(doclen_list[i, j].item())
    #             end_pos = start_pos + doclen
                
    #             sample_input_embeds.append(full_repr[i, start_pos:end_pos])    # (doc_len, D_llm)
    #             sample_input_embeds.append(query_hidden_states[flat_idx])    # (S, D_llm)
    #             start_pos = end_pos
    #         sample_input_embeds.append(full_repr[i, start_pos:effective_len])
    #         sample_input_embeds = torch.cat(sample_input_embeds, dim=0)
            
    #         # safe max length
    #         if sample_input_embeds.shape[0] > self.cfg.max_interleaving_len:
    #             sample_input_embeds = sample_input_embeds[:self.cfg.max_interleaving_len]
            
    #         seq_len = sample_input_embeds.shape[0]
    #         target_label = torch.full((seq_len,), -100, dtype=torch.long, device=sample_input_embeds.device)
    #         target_ids = target_input_ids[i, effective_len - a_len:effective_len]   # (a_len,)
            
    #         valid_a_len = min(a_len, seq_len)
    #         target_label[-valid_a_len:] = target_ids[-valid_a_len:]
    #         labels_list.append(target_label)
    #         input_embeds_list.append(sample_input_embeds)
    #         attention_mask_list.append(torch.ones(seq_len, dtype=torch.long, device=sample_input_embeds.device))

    #     # Pad manually
    #     padded_inputs_embeds = pad_sequence(input_embeds_list, batch_first=True, padding_value=0.0)
    #     padded_inputs_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    #     padded_target_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    #     # Generate logits and compute loss
    #     # GPU MEMORY BOTTLENECK!! (~13GB for 10 docs)
    #     # TODO: Document-Isolated Masking
    #     batch_logits = []
    #     batch_labels = []
    #     added_kwargs = {}
    #     if getattr(self.cfg, "do_isolate", False):
    #         added_kwargs["document_ids"] = self.convert_document_ids_to_tensor(
    #             padded_inputs_embeds, doclen_list, query_hidden_states, question_attention_mask)

    #     outputs = self.llm.model(
    #         inputs_embeds=padded_inputs_embeds,
    #         attention_mask=padded_inputs_attention_mask,
    #         use_cache=False,
    #         **added_kwargs  # This is for custom attention masking
    #     )
    #     hidden_states = outputs.last_hidden_state  # (B, L_total, D_llm)

    #     gen_loss = 0.0
    #     for i in range(B):
    #         a_len = int(a_len_list[i].item())
    #         effective_len = int(padded_inputs_attention_mask[i].sum().item())
    #         start_pos = effective_len - a_len
    #         end_pos = effective_len

    #         ans_hidden = hidden_states[i, start_pos-1:end_pos-1]    # (a_len, D_llm)
    #         ans_labels = padded_target_labels[i, start_pos:end_pos]

    #         ans_logits = self.llm.lm_head(ans_hidden)   # (a_len, V)
    #         sample_loss = self.gen_loss_fn(ans_logits, ans_labels)

    #         if return_logits:
    #             batch_logits.append(ans_logits.argmax(dim=-1).detach().cpu())
    #             batch_labels.append(ans_labels.detach().cpu())

    #         gen_loss += sample_loss
    #     gen_loss = gen_loss / B

    #     outputs = (gen_loss,)
    #     if return_logits:
    #         outputs += (batch_logits, batch_labels)
        
    #     return outputs
    
    def transform_scores_oracle(self, scores_oracle):
        if self.cfg.score_transform is None:
            return self.scaling_factor * scores_oracle
        elif self.cfg.score_transform == "sigmoid":
            return torch.sigmoid(scores_oracle / self.T)
        elif self.cfg.score_transform == "tanh":
            return torch.tanh(scores_oracle / self.T)
        else:
            raise ValueError(f"Unknown score transform: {self.cfg.score_transform}")

    def training_step(self, batch, batch_idx):
        llm_repr = self.forward(batch)
        scores_hat, query_hidden_states = self.caformer_clf(llm_repr, batch["source_attention_mask"],
                                                            batch["question_ids"], batch["question_attention_mask"])
        del llm_repr

        use_precompute_table = batch["scores_oracle"] is not None
        if not use_precompute_table:
            doc_repr = self.get_doc_repr(batch).detach()
            # Compute the gradient of the oracle loss and target score
            loss_oracle, loss_gradients = self.compute_oracle_loss_gradients(batch)
            scores_oracle = self.compute_oracle_scores(loss_gradients, doc_repr, batch["doclen_list"])    # (B*K, 1)
            scores_oracle = scores_oracle.detach()
            del loss_gradients, doc_repr
        else:
            scores_oracle = batch["scores_oracle"].reshape(-1, 1)   # (B*K, 1)
            scores_oracle = scores_oracle.to(scores_hat.device)
            scores_oracle = self.transform_scores_oracle(scores_oracle)

        # Guide loss
        scores_hat = scores_hat.view(-1)
        scores_oracle = scores_oracle.view(-1)
        guide_loss = self.guide_loss_fn(scores_hat, scores_oracle)
        # Generation loss
        gen_loss = self.compute_interleaving_loss(batch, query_hidden_states)[0]
        loss = self.lmbda * guide_loss + gen_loss
        
        if not use_precompute_table:
            self.log("train/oracle_loss", loss_oracle, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/guide_loss", guide_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/gen_loss", gen_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        llm_repr = self.forward(batch)
        scores_hat, query_hidden_states = self.caformer_clf(llm_repr, batch["source_attention_mask"],
                                                            batch["question_ids"], batch["question_attention_mask"])
        del llm_repr

        # Compute the gradient of the oracle loss and target score
        # Temporarily enable gradient tracking
        use_precompute_table = batch["scores_oracle"] is not None
        if not use_precompute_table:
            doc_repr = self.get_doc_repr(batch).detach()
            with torch.enable_grad():
                loss_oracle, loss_gradients = self.compute_oracle_loss_gradients(batch)
            scores_oracle = self.compute_oracle_scores(loss_gradients, doc_repr, batch["doclen_list"])    # (B*K, 1)
            scores_oracle = scores_oracle.detach()
            del loss_gradients, doc_repr
        else:
            scores_oracle = batch["scores_oracle"].reshape(-1, 1)   # (B*K, 1)
            scores_oracle = scores_oracle.to(scores_hat.device)
            scores_oracle = self.transform_scores_oracle(scores_oracle)

        # Guide loss
        scores_hat = scores_hat.view(-1)
        scores_oracle = scores_oracle.view(-1)
        guide_loss = self.guide_loss_fn(scores_hat, scores_oracle)
        # Generation loss
        gen_loss, batch_logits, batch_labels = self.compute_interleaving_loss(batch, query_hidden_states, return_logits=True)
        loss = self.lmbda * guide_loss + gen_loss
        
        self.val_logits.extend(batch_logits)
        self.val_labels.extend(batch_labels)

        if not use_precompute_table:
            self.log("valid/oracle_loss", loss_oracle, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/guide_loss", guide_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/gen_loss", gen_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/loss", loss, on_step=False, on_epoch=True, sync_dist=True)


    def on_validation_epoch_end(self):
        total_em = 0.0
        total_f1 = 0.0
        total_count = 0
        
        # Compute EM, F1 based on val_logits and val_labels
        for logits, labels in zip(self.val_logits, self.val_labels):
            pred = self.llm_tokenizer.decode(logits, skip_special_tokens=True)
            answer = self.llm_tokenizer.decode(labels, skip_special_tokens=True)
            gold_answers = parse_reference_answer(answer)
            metrics = compute_metrics(pred, gold_answers)
            
            total_em += float(metrics.soft_em)
            total_f1 += float(metrics.f1)
            total_count += 1

        avg_em = total_em / total_count if total_count > 0 else 0.0
        avg_f1 = total_f1 / total_count if total_count > 0 else 0.0
        
        self.log("valid/EM", avg_em, sync_dist=True)
        self.log("valid/F1", avg_f1, sync_dist=True)

        self.val_logits.clear()
        self.val_labels.clear()
        
    def on_save_checkpoint(self, checkpoint):
        # save only CAFormer classifier weights
        state_dict = checkpoint["state_dict"]
        caformer_clf_state_dict = {
            k: v for k, v in state_dict.items() if "caformer_clf" in k
        }
        checkpoint["state_dict"] = caformer_clf_state_dict

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.caformer_clf.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, fused=True)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }