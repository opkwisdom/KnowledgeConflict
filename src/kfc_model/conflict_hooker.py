import torch
import torch.nn as nn

class ConflictHooker:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.cache = {}

        self.attn_target, self.ffn_target, self.norm_target = self._resolve_target_modules()

    def _resolve_target_modules(self):
        """
        Return module names for hooking based on the model structure.
        """
        if hasattr(self.model, "model"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        elif hasattr(self.model, "language_model"): # Gemma-3
            layers = self.model.language_model.model.layers
        else:
            raise ValueError(f"Unknown Model Structure: {type(self.model)}")

        sample_layer = layers[0]
        attn_name = None
        ffn_name = None
        norm_name = None

        if hasattr(sample_layer, "self_attn"):
            if hasattr(sample_layer.self_attn, "o_proj"):
                attn_name = "self_attn.o_proj"
            elif hasattr(sample_layer.self_attn, "dense"):
                attn_name = "self_attn.dense"

        if hasattr(sample_layer, "mlp"):
            if hasattr(sample_layer.mlp, "down_proj"):
                ffn_name = "mlp.down_proj"
            elif hasattr(sample_layer.mlp, "dense_4h_to_h"):
                ffn_name = "mlp.dense_4h_to_h"
            elif hasattr(sample_layer.mlp, "c_proj"):
                ffn_name = "mlp.c_proj"
        
        if hasattr(sample_layer, "input_layernorm"):
            norm_name = "input_layernorm"
        elif hasattr(sample_layer, "ln_1"):
            norm_name = "ln_1"
        elif hasattr(sample_layer, "ln_attn"):
            norm_name = "ln_attn"

        if not all([attn_name, ffn_name, norm_name]):
             raise ValueError("Failed to resolve Attn/FFN/Norm modules automatically.")
             
        return attn_name, ffn_name, norm_name

    def _get_hook_fn(self, key):
        def hook(module, input, output):
            # output: (B, S, D)
            self.cache[key] = output[:, -1, :].detach().cpu()
        return hook

    def remove(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def register(self):
        self.remove()
        self.cache.clear()

        if hasattr(self.model, "model"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            layers = self.model.language_model.model.layers

        for i, layer in enumerate(layers):
            attn_module = self._get_submodule(layer, self.attn_target)
            ffn_module = self._get_submodule(layer, self.ffn_target)
            norm_module = self._get_submodule(layer, self.norm_target)

            self.hooks.append(attn_module.register_forward_hook(self._get_hook_fn(f"attn_{i}")))
            self.hooks.append(ffn_module.register_forward_hook(self._get_hook_fn(f"ffn_{i}")))
            self.hooks.append(norm_module.register_forward_hook(self._get_hook_fn(f"norm_{i}")))
    
    def _get_submodule(self, layer, path):
        parts = path.split(".")
        module = layer
        for part in parts:
            module = getattr(module, part)
        return module
    
    def get_cache(self, layer_idx: int):
        return self.cache.get(f"attn_{layer_idx}"), self.cache.get(f"ffn_{layer_idx}"), self.cache.get(f"norm_{layer_idx}")

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()