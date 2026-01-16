import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from typing import List
from tqdm import tqdm
from dataclasses import asdict, dataclass
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from kfc_model import SimpleIrrDetector
from utils import set_seed, setup_logger

@dataclass
class TestEx:
    id: int
    question: str
    a_internal: str
    answers: List[str]
    ctx_idx: int
    ctx_rel: str
    kv_cache_score: torch.Tensor

### ========== Model and training arguments ========== ###
model_args = {
    "num_layers": 32,
    "top_k": 100
}

BATCH_SIZE = 128
TEST_SIZE = 0.2
SEED = 42
EPOCHS = 30
LEARNING_RATE = 1e-4
DATA_PATH = "experiments/irr_detector_data/train/kv_cache_scores_data.pkl"
OUTPUT_DIR = "src/irr_det_results"
### ========== Model and training arguments ========== ###

def main():
    set_seed(SEED)
    logger = setup_logger("KFC-Irr-Detector-Training", OUTPUT_DIR)

    # Initialize wandb
    run = wandb.init(project="Knowledge-Fusion-Core")
    config = run.config
    logger.info("WandB initialized.")

    # Load dataset
    with open(DATA_PATH, "rb") as f:
        data_dict = pickle.load(f)
    X, y = data_dict["kv_cache_scores"], data_dict["labels"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    logger.info(f"Dataset loaded with {len(X_train)} training and {len(X_test)} testing samples.")

    train_loader = DataLoader(
        list(zip(X_train, y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        list(zip(X_test, y_test)),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialize and log the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    irr_detector = SimpleIrrDetector(**model_args).to(device)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    optimizer = torch.optim.AdamW(irr_detector.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float('inf')
    logger.info("Starting training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
        ### Training Loop ###
        irr_detector.train()
        total_loss = 0.0

        for batch in train_loader:
            scores, labels = batch
            scores = scores.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = irr_detector(scores)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]

            wandb.log({
                "train/step_loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/epoch": epoch
            })
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        ### Evaluation Loop ###
        irr_detector.eval()
        correct, total = 0, 0
        val_loss = 0.0
        for batch in test_loader:
            scores, labels = batch
            scores = scores.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = irr_detector(scores)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_loss += criterion(logits, labels.float()).item()
        
        accuracy = correct / total
        avg_val_loss = val_loss / len(test_loader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Test Accuracy: {accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")

        wandb.log({
            "test/accuracy": accuracy,
            "test/val_loss": avg_val_loss,
            "test/epoch": epoch
        })
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(irr_detector.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            logger.info(f"Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()