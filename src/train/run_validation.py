from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from tqdm import tqdm
import logging
import sys
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, proj_root)

from src.train.datamodule import *
from src.train.model import *
from src.utils import load_config, setup_logger

@dataclass
class FeatureExample:
    features: torch.Tensor
    label: int

logger = logging.getLogger(__name__)

DATAMODULE_DICT = {
    "features": FeatureDataModule,
    "fusion": FusionDataModule,
}
MODEL_DICT = {
    "features": (ConflictFeatureDetector, ConflictFeatureDetectorModule),
    "fusion": (ConflictFusionDetector, ConflictFusionDetectorModule),
}

# @dataclass
# class FeatureExample:
#     features: torch.Tensor
#     h_mid: torch.Tensor
#     h_last: torch.Tensor
#     logit_diff: torch.Tensor
#     label: int

def get_best_checkpoint(ckpt_dir, monitor='val_loss'):
    best_ckpt = None
    best_score = float('inf')
    for filename in os.listdir(ckpt_dir):
        if filename.endswith('.ckpt') and monitor in filename:
            parts = filename.split(f'{monitor}=')[1]
            score_str = parts.split('.ckpt')[0]
            try:
                score = float(score_str)
                if score < best_score:
                    best_score = score
                    best_ckpt = os.path.join(ckpt_dir, filename)
            except ValueError:
                continue
    return best_ckpt

def inference(model, datamodule, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    datamodule.setup(stage='validate')
    dataloader = datamodule.val_dataloader()
    
    all_preds = []
    all_targets = []
    all_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            preds = preds.cpu().tolist()
            labels = labels.cpu().tolist()
            inputs = inputs.cpu()
            all_preds.extend(preds)
            all_targets.extend(labels)
            all_features.extend(inputs)

    preds_tensor = torch.tensor(all_preds)
    targets_tensor = torch.tensor(all_targets)
    features_tensor = torch.stack(all_features)

    save_data = {
        "features": features_tensor,    # (N, L, H, F)
        "preds": preds_tensor,          # (N,)
        "targets": targets_tensor       # (N,)
    }
    return save_data

def analyze_results(save_path):
    logger = logging.getLogger(__name__)
    results = torch.load(save_path, map_location="cpu", weights_only=False)
    output_dir = os.path.dirname(save_path)

    features = results["features"]  # (N, L, H, F)
    preds = results["preds"]        # (N,)
    targets = results["targets"]    # (N,)

    features_ent = features[..., 0]
    features_sum = features[..., 1]
    
    # Analyze results by cases
    masks = {
        "TP (Correct Pos)": (preds == 1) & (targets == 1),
        "TN (Correct Neg)": (preds == 0) & (targets == 0),
        "FP (False Alarm)": (preds == 1) & (targets == 0),
        "FN (Missed)":      (preds == 0) & (targets == 1),
    }

    def plot_layer_trend(feature_tensor, feature_name):
        plt.figure(figsize=(10, 6))
        
        for label, mask in masks.items():
            if mask.sum() == 0: continue
            
            # (N_subset, L, H) -> (N_subset, L) -> (L,)
            subset = feature_tensor[mask]
            layer_avg = subset.float().mean(dim=2).mean(dim=0).numpy()
            layer_std = subset.float().mean(dim=2).std(dim=0).numpy()
            
            x = range(len(layer_avg))
            plt.plot(x, layer_avg, label=label, linewidth=2)
            plt.fill_between(x, layer_avg - layer_std, layer_avg + layer_std, alpha=0.1)
            
        plt.title(f"Layer-wise Average {feature_name} Trend")
        plt.xlabel("Layer Index")
        plt.ylabel(f"Average {feature_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = f"{output_dir}/trend_{feature_name.lower()}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved trend analysis to {output_path}")

    def plot_mean_std_heatmaps(feature_tensor, feature_name):
        valid_keys = [k for k, m in masks.items() if m.sum() > 0]
        # Mean, Std heatmaps
        fig, axes = plt.subplots(2, len(valid_keys), figsize=(5 * len(valid_keys), 10))
        
        global_mean_min = feature_tensor.mean(dim=0).min().item()
        global_mean_max = feature_tensor.mean(dim=0).max().item()
        global_std_max = feature_tensor.std(dim=0).max().item()

        if len(valid_keys) == 1: axes = [axes]

        for i, key in enumerate(valid_keys):
            mask = masks[key]
            subset = feature_tensor[mask].float()
            # (N_sub, L, H) -> (L, H)
            mean_map = subset.mean(dim=0).numpy()
            std_map = subset.std(dim=0).numpy()

            # Row 1: Mean
            ax_mean = axes[0, i]
            sns.heatmap(mean_map, ax=ax_mean, cmap="viridis", vmin=global_mean_min, vmax=global_mean_max)
            ax_mean.set_title(f"{key} ({mask.sum()} samples)")
            ax_mean.set_ylabel("Layer") if i == 0 else ax_mean.set_ylabel("")
            ax_mean.set_xticks([])
            
            # Row 2: Std
            ax_std = axes[1, i]
            sns.heatmap(std_map, ax=ax_std, cmap="magma", vmin=0, vmax=global_std_max)
            ax_std.set_xlabel("Head Index")
            ax_std.set_ylabel("Layer") if i == 0 else ax_std.set_ylabel("")

        plt.suptitle(f"{feature_name} Mean vs Std Analysis", y=1.02)
        plt.tight_layout()

        output_path = f"{output_dir}/heatmap_{feature_name.lower()}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved heatmap analysis to {output_path}")

    logger.info("\n=== 1. Layer-wise Trend Analysis ===")
    plot_layer_trend(features_sum, "sum")
    plot_layer_trend(features_ent, "entropy")
    
    logger.info("\n=== 2. (L, H) Heatmap Analysis ===")
    plot_mean_std_heatmaps(features_sum, "sum")
    plot_mean_std_heatmaps(features_ent, "entropy")



def main():
    cfg = load_config()
    seed_everything(cfg.seed)
    setup_logger(f"validation_{cfg.exp_type}", os.path.join(cfg.output_dir, cfg.exp_type))
    logger = logging.getLogger(__name__)

    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(cfg))

    save_path = os.path.join(cfg.output_dir, cfg.exp_type, f"inference_results.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Inference
    if not os.listdir(os.path.dirname(save_path)):
        # Load datamodule
        datamodule = DATAMODULE_DICT[cfg.exp_type](cfg.data)

        # Load model
        model_class, module_class = MODEL_DICT[cfg.exp_type]
        model = model_class(cfg.model, cfg.model.n_class)

        ckpt_dir = os.path.join(cfg.trained_dir, cfg.exp_type)
        best_checkpoint = get_best_checkpoint(ckpt_dir)
        logger.info(f"Best checkpoint found at: {best_checkpoint}")
        lightning_module = module_class.load_from_checkpoint(best_checkpoint, cfg=cfg.train, model=model, weights_only=False)

        model = lightning_module.model
        model.eval()
    
        logger.info("Starting inference...")
        inference_result = inference(model, datamodule, cfg)
        torch.save(inference_result, save_path)
    
    # Analysis
    logger.info("Starting analysis...")
    analyze_results(save_path)


if __name__ == "__main__":
    main()