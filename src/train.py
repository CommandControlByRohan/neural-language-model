# train.py
import os
import json
import math
import time
import argparse
import logging
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import LSTMLanguageModel

# -----------------------
# Argument parsing
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--drive_dir", type=str, default=".", help="Base folder (Colab Drive or local).")
parser.add_argument("--experiment", type=str, default="bestfit", help="Experiment name (used for saving).")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--hidden_size", type=int, default=None)
parser.add_argument("--embed_size", type=int, default=None)
parser.add_argument("--num_layers", type=int, default=None)
parser.add_argument("--dropout", type=float, default=None)
parser.add_argument("--embed_dropout", type=float, default=None)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--patience", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test and exit.")
args = parser.parse_args()

# -----------------------
# Experiment presets (Preset A = best-fit)
# -----------------------
# Default best-fit hyperparameters:
preset = {
    "embed_size": 256,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.45,
    "embed_dropout": 0.10,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "epochs": 15,
    "clip": 1.0,
    "patience": args.patience
}

# Apply CLI overrides (if provided)
if args.embed_size is not None: preset["embed_size"] = args.embed_size
if args.hidden_size is not None: preset["hidden_size"] = args.hidden_size
if args.num_layers is not None: preset["num_layers"] = args.num_layers
if args.dropout is not None: preset["dropout"] = args.dropout
if args.embed_dropout is not None: preset["embed_dropout"] = args.embed_dropout
if args.lr is not None: preset["lr"] = args.lr
if args.weight_decay is not None: preset["weight_decay"] = args.weight_decay
if args.batch_size is not None: preset["batch_size"] = args.batch_size
if args.epochs is not None: preset["epochs"] = args.epochs

# Smoke test shortcut
if args.smoke:
    preset["epochs"] = 1
    preset["batch_size"] = min(16, preset["batch_size"])

# -----------------------
# Repro & device
# -----------------------
random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Create experiment folder + logger
# -----------------------
timestamp = time.strftime("%Y%m%d-%H%M%S")
exp_name = f"{args.experiment}_{timestamp}"
exp_dir = os.path.join(args.drive_dir, "experiments", exp_name)
os.makedirs(exp_dir, exist_ok=True)

# Logger that logs to console and file
logger = logging.getLogger("train_logger")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
# console
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)
# file
fh = logging.FileHandler(os.path.join(exp_dir, f"train_{args.experiment}.log"))
fh.setFormatter(fmt)
logger.addHandler(fh)

logger.info(f"Experiment dir: {exp_dir}")
logger.info(f"Preset hyperparameters: {preset}")

# -----------------------
# Data loader
# -----------------------
def load_data(drive_dir):
    # Expect data_prep.py already run and saved train.pt/val.pt/test.pt and vocab.json
    train = torch.load(os.path.join(drive_dir, "train.pt"))
    val   = torch.load(os.path.join(drive_dir, "val.pt"))
    test  = torch.load(os.path.join(drive_dir, "test.pt"))
    with open(os.path.join(drive_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab["word2id"])
    train_loader = DataLoader(TensorDataset(train["x"], train["y"]), batch_size=preset["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val["x"], val["y"]), batch_size=preset["batch_size"],
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(TensorDataset(test["x"], test["y"]), batch_size=preset["batch_size"],
                             shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader, vocab_size

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total += loss.item() * x.size(0)
        count += x.size(0)
    avg = total / count
    return avg, math.exp(avg)

# -----------------------
# Training loop
# -----------------------
def train(drive_dir, exp_dir):
    train_loader, val_loader, test_loader, vocab_size = load_data(drive_dir)
    model = LSTMLanguageModel(vocab_size=vocab_size,
                              embed_size=preset["embed_size"],
                              hidden_size=preset["hidden_size"],
                              num_layers=preset["num_layers"],
                              dropout=preset["dropout"],
                              embed_dropout=preset["embed_dropout"]).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=preset["lr"], weight_decay=preset["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val = float("inf")
    stalled = 0
    patience = preset["patience"]
    train_losses, val_losses = [], []

    for epoch in range(1, preset["epochs"] + 1):
        model.train()
        total_train = 0.0
        for batch_idx, (x, y) in enumerate(train_loader): # Added enumerate to get batch_idx
            if batch_idx % 100 == 0: # Print every 100 batches
                logger.info(f"Epoch {epoch}/{preset['epochs']} - Processing batch {batch_idx}/{len(train_loader)}")
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train += loss.item() * x.size(0)
        avg_train = total_train / len(train_loader.dataset)
        val_loss, val_ppl = evaluate(model, val_loader, criterion)

        train_losses.append(avg_train); val_losses.append(val_loss)
        logger.info(f"Epoch {epoch}/{preset['epochs']} | Train loss: {avg_train:.4f} | Val loss: {val_loss:.4f} | Val ppl: {val_ppl:.2f}")

        # scheduler step (use val_loss)
        scheduler.step(val_loss)
        logger.info(f"Scheduler LR: {optimizer.param_groups[0]['lr']}")

        # early stopping + checkpoint
        if val_loss < best_val:
            best_val = val_loss
            stalled = 0
            best_path = os.path.join(exp_dir, f"best_model_{args.experiment}.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(f" Saved best model to {best_path}")
        else:
            stalled += 1
            if stalled > patience:
                logger.info(f"No improvement for {patience} epochs -> early stopping")
                break

    # load best and evaluate test
    best_path = os.path.join(exp_dir, f"best_model_{args.experiment}.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        test_loss, test_ppl = evaluate(model, test_loader, criterion)
        logger.info(f"Test loss: {test_loss:.4f} | Test perplexity: {test_ppl:.2f}")
    else:
        logger.warning("No best model saved; skipping final test eval.")
        test_loss, test_ppl = None, None

    # Save losses & metadata
    torch.save({"train_losses": train_losses, "val_losses": val_losses}, os.path.join(exp_dir, f"losses_{args.experiment}.pth"))
    meta = {
        "preset": preset,
        "seed": args.seed,
        "final_test_loss": test_loss,
        "final_test_ppl": test_ppl
    }
    with open(os.path.join(exp_dir, f"meta_{args.experiment}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Experiment artifacts saved to {exp_dir}")

if __name__ == "__main__":
    train(args.drive_dir, exp_dir)

