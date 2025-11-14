# train_underfit.py
# Small, standalone script to run the "underfitting" experiment.
# Usage (from project folder containing model.py, train.pt, val.pt, test.pt, vocab.json):
#   python train_underfit.py --drive_dir "." --experiment underfit --epochs 5

import os
import json
import time
import argparse
import logging
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import LSTMLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("--drive_dir", type=str, default=".", help="Project folder with train.pt/val.pt/test.pt/vocab.json")
parser.add_argument("--experiment", type=str, default="underfit", help="experiment name")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke", action="store_true", help="1-epoch smoke test")
args = parser.parse_args()

# Underfit preset (small model, little/no regularization, short training)
preset = {
    "embed_size": 128,
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.0,
    "embed_dropout": 0.0,
    "lr": args.lr,
    "weight_decay": 0.0,
    "batch_size": args.batch_size,
    "epochs": 1 if args.smoke else args.epochs,
    "clip": 1.0
}

# reproducibility & device
random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create experiment dir + logger
timestamp = time.strftime("%Y%m%d-%H%M%S")
exp_name = f"{args.experiment}_{timestamp}"
exp_dir = os.path.join(args.drive_dir, "experiments", exp_name)
os.makedirs(exp_dir, exist_ok=True)

logger = logging.getLogger("underfit")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
ch = logging.StreamHandler()
ch.setFormatter(fmt); logger.addHandler(ch)
fh = logging.FileHandler(os.path.join(exp_dir, f"train_{args.experiment}.log"))
fh.setFormatter(fmt); logger.addHandler(fh)

logger.info(f"Experiment dir: {exp_dir}")
logger.info(f"Preset hyperparameters: {preset}")

def load_data(drive_dir):
    train = torch.load(os.path.join(drive_dir, "train.pt"))
    val   = torch.load(os.path.join(drive_dir, "val.pt"))
    test  = torch.load(os.path.join(drive_dir, "test.pt"))
    with open(os.path.join(drive_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab["word2id"])
    train_loader = DataLoader(TensorDataset(train["x"], train["y"]), batch_size=preset["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val["x"], val["y"]), batch_size=preset["batch_size"], shuffle=False, drop_last=False)
    test_loader = DataLoader(TensorDataset(test["x"], test["y"]), batch_size=preset["batch_size"], shuffle=False, drop_last=False)
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

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, preset["epochs"] + 1):
        model.train()
        total_train = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx % 200 == 0:
                logger.info(f"Epoch {epoch}/{preset['epochs']} - Processing batch {batch_idx}/{len(train_loader)}")
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), preset["clip"])
            optimizer.step()
            total_train += loss.item() * x.size(0)
        avg_train = total_train / len(train_loader.dataset)
        val_loss, val_ppl = evaluate(model, val_loader, criterion)
        train_losses.append(avg_train); val_losses.append(val_loss)
        logger.info(f"Epoch {epoch}/{preset['epochs']} | Train loss: {avg_train:.4f} | Val loss: {val_loss:.4f} | Val ppl: {val_ppl:.2f}")
        # save best anyway (but underfit run is short)
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(exp_dir, f"best_model_{args.experiment}.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(f" Saved best model to {best_path}")

    # final evaluation on test (load best if exists)
    best_path = os.path.join(exp_dir, f"best_model_{args.experiment}.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
    test_loss, test_ppl = evaluate(model, test_loader, criterion)
    logger.info(f"Test loss: {test_loss:.4f} | Test perplexity: {test_ppl:.2f}")

    torch.save({"train_losses": train_losses, "val_losses": val_losses}, os.path.join(exp_dir, f"losses_{args.experiment}.pth"))
    meta = {"preset": preset, "seed": args.seed, "final_test_loss": test_loss, "final_test_ppl": test_ppl}
    with open(os.path.join(exp_dir, f"meta_{args.experiment}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Experiment artifacts saved to {exp_dir}")

if __name__ == "__main__":
    train(args.drive_dir, exp_dir)
