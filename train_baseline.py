# ================================================================
# UniOcc Baseline Training + Plotting
# ---------------------------------------------------------------
# - Loads NuScenes-via-Occ3D-2Hz-mini
# - Uses SimpleOccNet (3D U-Net baseline)
# - Binary occupancy forecasting (occupied vs free)
# - Logs Loss & IoU
# - Saves CSV logs
# - Generates Matplotlib plots automatically
# ================================================================

import os
import argparse
import time
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from uniocc_dataset import UniOcc
from BASELINE_model.simple3d_unet import SimpleOccNet


FREE_LABEL = 10  # UniOcc "free space" label


# ================================================================
# DEVICE SELECTION
# ================================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ================================================================
# DATASET LOADING / DATALOADER
# ================================================================
def build_dataloaders(data_root, obs_len, fut_len, batch_size, num_workers, val_split):
    dataset = UniOcc(
        data_root=data_root,
        obs_len=obs_len,
        fut_len=fut_len,
    )

    n_total = len(dataset)
    n_val = max(1, int(val_split * n_total))
    n_train = n_total - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=UniOcc.collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=UniOcc.collate_fn,
    )

    return train_loader, val_loader


# ================================================================
# LABEL PROCESSING
# ================================================================
def occ_to_binary(t, free_label=FREE_LABEL):
    """
    Convert semantic grid into:
        1 = occupied
        0 = free
    """
    return (t != free_label).float()


# ================================================================
# METRIC: IoU
# ================================================================
def compute_batch_iou(logits, target_binary, threshold=0.5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        preds = preds.view(-1)
        target = target_binary.view(-1)

        intersection = torch.sum((preds == 1) & (target == 1)).item()
        union = torch.sum((preds == 1) | (target == 1)).item()

        if union == 0:
            return 1.0
        return intersection / union


# ================================================================
# TRAIN EPOCH
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, log_interval):

    model.train()
    running_loss = 0
    running_iou = 0
    n_batches = 0

    start = time.time()

    for batch_idx, batch in enumerate(loader):
        obs_occ = batch["obs_occ_labels"].to(device).float()
        fut_occ = batch["fut_occ_labels"].to(device)

        fut_binary = occ_to_binary(fut_occ)

        optimizer.zero_grad()

        pred_logits = model(obs_occ)
        loss = criterion(pred_logits, fut_binary)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        iou = compute_batch_iou(pred_logits, fut_binary)

        running_loss += loss.item()
        running_iou += iou
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"[Train] Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                f"| Loss: {running_loss/n_batches:.4f} | IoU: {running_iou/n_batches:.4f}"
            )

    avg_loss = running_loss / max(n_batches, 1)
    avg_iou = running_iou / max(n_batches, 1)

    print(f"[Train] Epoch {epoch} done in {time.time()-start:.1f}s | Loss {avg_loss:.4f} | IoU {avg_iou:.4f}")

    return avg_loss, avg_iou


# ================================================================
# VALIDATION EPOCH
# ================================================================
def validate_one_epoch(model, loader, criterion, device, epoch):

    model.eval()
    running_loss = 0
    running_iou = 0
    n_batches = 0

    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            obs_occ = batch["obs_occ_labels"].to(device).float()
            fut_occ = batch["fut_occ_labels"].to(device)

            fut_binary = occ_to_binary(fut_occ)

            pred_logits = model(obs_occ)
            loss = criterion(pred_logits, fut_binary)
            iou = compute_batch_iou(pred_logits, fut_binary)

            running_loss += loss.item()
            running_iou += iou
            n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_iou = running_iou / max(n_batches, 1)

    print(f"[Val] Epoch {epoch} done in {time.time()-start:.1f}s | Loss {avg_loss:.4f} | IoU {avg_iou:.4f}")

    return avg_loss, avg_iou


# ================================================================
# CHECKPOINT SAVE
# ================================================================
def save_checkpoint(model, optimizer, epoch, save_dir, tag="best"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"uniocc_baseline_{tag}.pth")
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        path,
    )
    print(f"✓ Saved checkpoint: {path}")


# ================================================================
# PLOTTING UTILITIES
# ================================================================
def plot_curve(values, title, ylabel, save_path):
    plt.figure(figsize=(7,5))
    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Saved plot: {save_path}")


# ================================================================
# MAIN
# ================================================================
def main():

    parser = argparse.ArgumentParser("UniOcc Baseline Training")
    parser.add_argument("--data-root", type=str, default="datasets/NuScenes-via-Occ3D-2Hz-mini")
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--fut-len", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(
        args.data_root,
        args.obs_len,
        args.fut_len,
        args.batch_size,
        args.num_workers,
        args.val_split,
    )

    model = SimpleOccNet(obs_len=args.obs_len, fut_len=args.fut_len).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup log dirs
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []

    best_val_iou = -1

    # ---------------- Training Loop ----------------
    for epoch in range(1, args.epochs + 1):

        tr_loss, tr_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.log_interval
        )
        va_loss, va_iou = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        train_losses.append(tr_loss)
        train_ious.append(tr_iou)
        val_losses.append(va_loss)
        val_ious.append(va_iou)

        # Save best
        if va_iou > best_val_iou:
            best_val_iou = va_iou
            save_checkpoint(model, optimizer, epoch, args.save_dir, "best")

    # Save final
    save_checkpoint(model, optimizer, args.epochs, args.save_dir, "last")

    # ---------------- Save CSV Logs ----------------
    with open("logs/train_log.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_iou"])
        for i in range(len(train_losses)):
            writer.writerow([i+1, train_losses[i], train_ious[i]])

    with open("logs/val_log.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_loss", "val_iou"])
        for i in range(len(val_losses)):
            writer.writerow([i+1, val_losses[i], val_ious[i]])

    print("✓ CSV logs saved")

    # ---------------- Generate Plots ----------------
    plot_curve(train_losses, "Train Loss", "Loss", "plots/train_loss_curve.png")
    plot_curve(train_ious, "Train IoU", "IoU", "plots/train_iou_curve.png")
    plot_curve(val_losses, "Val Loss", "Loss", "plots/val_loss_curve.png")
    plot_curve(val_ious, "Val IoU", "IoU", "plots/val_iou_curve.png")

    print("\n Training complete! All plots + logs saved.")

if __name__ == "__main__":
    main()
