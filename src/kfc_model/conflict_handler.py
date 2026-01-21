import logging
import json
from omegaconf import DictConfig
from typing import List, Tuple, Optional, Dict
from transformers import LlamaConfig, AutoConfig

from .conflict_resources import *

logger = logging.getLogger(__name__)

class ConflictConfigHandler:
    def __init__(self, config: DictConfig, model_config: AutoConfig) -> None:
        self.config = config
        self.model_config = model_config
        self.critical_map, self.normal_map = self._setup_indices()
        self.control_cache_stats, self.selected_d = self._load_control_stats()

    def _parse_critical_indices(self, critical_indices: Optional[List[str]]
        ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        NUM_HEADS = getattr(self.model_config, "num_key_value_heads", None)
        NUM_LAYERS = self.model_config.num_hidden_layers

        critical_map: Dict[int, List[int]] = {}
        normal_map: Dict[int, List[int]] = {}

        for item in critical_indices:
            parts = item.split('_')
            layer_idx = int(parts[0])

            if layer_idx not in critical_map:
                critical_map[layer_idx] = []
            
            if len(parts) == 1:
                critical_map[layer_idx] = list(range(NUM_HEADS))
            else:
                head_idx = int(parts[1])
                critical_map[layer_idx].append(head_idx)
        
        for layer_idx in range(NUM_LAYERS):
            all_heads_in_layer = set(range(NUM_HEADS))
            critical_heads = critical_map.get(layer_idx, set())

            normal_heads = all_heads_in_layer - set(critical_heads)
            if normal_heads:
                normal_map[layer_idx] = list(normal_heads)
        
        return critical_map, normal_map
    
    def _setup_indices(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        critical_indices = None
        if isinstance(self.model_config, LlamaConfig):
            critical_indices = LLAMA_CRITICAL_INDICES
            self.selected_layers = LLAMA_CRITICAL_INDICES

        if critical_indices:
            critical_map, normal_map = self._parse_critical_indices(critical_indices)
        else:
            critical_map, normal_map = {}, {}
            logger.warning("No critical indices found for the model.")

        return critical_map, normal_map


    def _load_control_stats(self) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
        # control cache stats
        control_metadata = self.config.model.get("control_metadata", None)
        control_cache_path = control_metadata.get("stats_path", None)
        control_layer_info_path = control_metadata.get("layer_info_path", None)

        if control_cache_path:
            with open(control_cache_path, "r") as f:
                _control_cache_stats = json.load(f)
            _control_stats = _control_cache_stats["control_stat"]
            control_cache_stats = {
                int(idx): {
                    "mean": _control_stats[int(idx)]["mean"],
                    "std": _control_stats[int(idx)]["std"],
                }
                for idx in self.selected_layers
            }
        else:
            control_cache_stats = None
            logger.warning("No control cache stats path provided.")

        if control_layer_info_path:
            with open(control_layer_info_path, "r") as f:
                _conflict_layer_info = json.load(f)
            selected_d = {
                int(idx): _conflict_layer_info["sorted_scores"]["mean_d"][i]
                for i, idx in enumerate(self.selected_layers)
            }
        else:
            selected_d = None
            logger.warning("No conflict layer info path provided.")

        return control_cache_stats, selected_d