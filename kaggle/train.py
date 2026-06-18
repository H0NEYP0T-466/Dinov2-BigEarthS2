#!/usr/bin/env python3
"""
Dinov2-BigEarthS2 — Training Pipeline.

Single-file script that orchestrates the full training pipeline.
Just run:

    pip install -r train_requirements.txt
    python train.py

Everything is here: config, dataset streaming, model, training loop,
checkpointing, curves, history export. No notebook needed.

The architecture, 43-class list, and checkpoint format exactly match the
FastAPI backend in backend/app/models/dinov2.py and backend/app/config.py.
The trained model_best.pth drops straight into backend/checkpoints/.

Works on:
    - Kaggle (GPU P100 or T4 x2, Internet ON, Persistence Files)
    - Any machine with CUDA and ~16 GB VRAM (P100-class)
    - CPU (slow but functional — set use_amp=False)
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()

# ---------------------------------------------------------------------------
# Configuration  (edit these to taste)
# ---------------------------------------------------------------------------
CFG = {
    # --- Model ---
    "backbone": "vit_base_patch14_dinov2.lvd142m",  # DINOv2-B, 86M, 768-d
    "backbone_dim": 768,
    "num_classes": 43,
    "head_dropout": 0.3,
    "image_size": 224,
    # --- Data ---
    "batch_size": 128 if N_GPUS >= 2 else 64,
    "num_workers": 4,
    "grad_accum_steps": 2,
    # --- Three-phase fine-tuning ---
    "phases": [
        {"name": "head", "epochs": 5, "lr": 1e-3, "unfreeze": "head"},
        {"name": "last_2_blocks", "epochs": 10, "lr": 1e-4, "unfreeze": "last_2_blocks"},
        {"name": "full", "epochs": 15, "lr": 1e-5, "unfreeze": "full"},
    ],
    # --- Optimiser / scheduler ---
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "sched_mode": "max",
    "sched_factor": 0.5,
    "sched_patience": 3,
    "sched_min_lr": 1e-7,
    # --- Early stopping ---
    "es_patience": 7,
    "es_min_delta": 0.001,
    # --- Metrics ---
    "threshold": 0.5,
    # --- Mixed precision ---
    "use_amp": torch.cuda.is_available(),
    # --- ImageNet normalisation (DINOv2 was pretrained with these) ---
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    # --- HuggingFace streaming dataset ---
    "hf_dataset": "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
    "hf_fallback": "GFM-Bench/BigEarthNet",
    "val_skip": 20000,  # first 20 k samples reserved for validation
    "val_steps_per_epoch": 300,
    # --- Output root (Kaggle-friendly, but works anywhere) ---
    "output_dir": "/kaggle/working" if Path("/kaggle/working").exists() else ".",
}

# ---------------------------------------------------------------------------
# Official BigEarthNet-S2 43-class nomenclature (torchgeo class_sets[43]).
# MUST mirror backend/app/config.py CLASSES_43 and src/types/index.ts CLASS_NAMES.
# ---------------------------------------------------------------------------
CLASSES_43 = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Road and rail networks and associated land",
    "Port areas",
    "Airports",
    "Mineral extraction sites",
    "Dump sites",
    "Construction sites",
    "Green urban areas",
    "Sport and leisure facilities",
    "Non-irrigated arable land",
    "Permanently irrigated land",
    "Rice fields",
    "Vineyards",
    "Fruit trees and berry plantations",
    "Olive groves",
    "Pastures",
    "Annual crops associated with permanent crops",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Bare rock",
    "Sparsely vegetated areas",
    "Burnt areas",
    "Inland marshes",
    "Peatbogs",
    "Salt marshes",
    "Salines",
    "Intertidal flats",
    "Water courses",
    "Water bodies",
    "Coastal lagoons",
    "Estuaries",
    "Sea and ocean",
]
assert len(CLASSES_43) == 43, f"Expected 43 classes, got {len(CLASSES_43)}"
CFG["classes"] = CLASSES_43


# ===================================================================
#  Dataset
# ===================================================================

def _normalize_sample(sample):
    """Convert one HF dataset row into (rgb_PIL, multi_hot_43_tensor)."""
    import timm  # noqa: already top-level, just for PIL
    from PIL import Image  # noqa

    # --- Image ---
    img = sample.get("image") or sample.get("img") or sample.get("s2")
    if img is None:
        raise KeyError(f"No image field in sample keys: {list(sample.keys())}")
    if not isinstance(img, Image.Image):
        raw = img["bytes"] if isinstance(img, dict) else img
        img = Image.open(io.BytesIO(raw))

    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 12:
        # torchgeo band order (B01,B02,B03,B04,...) → RGB = indices [3,2,1]
        rgb = arr[:, :, [3, 2, 1]]
        img = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))
    else:
        img = img.convert("RGB")

    # --- Labels → multi-hot 43 ---
    labels = sample.get("labels") or sample.get("label") or sample.get("label_multi_hot")
    if labels is None:
        raise KeyError(f"No label field in sample keys: {list(sample.keys())}")

    if isinstance(labels, list) and len(labels) == 43:
        multi_hot = torch.tensor(labels, dtype=torch.float32)
    elif isinstance(labels, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in labels) and max(labels) < 43:
        multi_hot = torch.zeros(43, dtype=torch.float32)
        multi_hot[list(labels)] = 1.0
    else:
        multi_hot = torch.tensor(labels, dtype=torch.float32).reshape(-1)[:43]
        if multi_hot.shape[0] < 43:
            multi_hot = F.pad(multi_hot, (0, 43 - multi_hot.shape[0]))

    assert multi_hot.shape[0] == 43, f"Bad label shape {multi_hot.shape}"
    return img, multi_hot


class _StreamingBuffer(Dataset):
    """Buffers a chunk of an HF IterableDataset for use with DataLoader."""

    def __init__(self, hf_stream, transform, buffer_size, is_train=True):
        self._iter = iter(hf_stream)
        self.transform = transform
        self.buffer_size = buffer_size
        self.is_train = is_train
        self._buf: list = []
        self._refill()

    def _refill(self):
        self._buf.clear()
        for _ in range(self.buffer_size):
            try:
                self._buf.append(_normalize_sample(next(self._iter)))
            except StopIteration:
                break
        if self.is_train:
            random.shuffle(self._buf)

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        if idx >= len(self._buf):
            self._refill()
            if idx >= len(self._buf):
                idx = 0
        img, label = self._buf[idx]
        return self.transform(img), label


def _build_dataloaders():
    """Stream BigEarthNet-S2 from HuggingFace and return (train_loader, val_loader)."""
    import torchvision.transforms as T
    from datasets import load_dataset  # noqa: local import — may not be installed until training

    train_tf = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=(0, 90)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=0.1, scale=(0.9, 1.1)),
        T.Resize((CFG["image_size"], CFG["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=CFG["imagenet_mean"], std=CFG["imagenet_std"]),
    ])
    val_tf = T.Compose([
        T.Resize((CFG["image_size"], CFG["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=CFG["imagenet_mean"], std=CFG["imagenet_std"]),
    ])

    # --- Try HF repos ---
    ds = None
    for repo in [CFG["hf_dataset"], CFG["hf_fallback"]]:
        try:
            print(f"[dataset] Attempting HF streaming from: {repo}")
            ds = load_dataset(repo, split="train", streaming=True)
            _normalize_sample(next(iter(ds)))  # probe
            print(f"[dataset] OK — streaming from {repo}")
            break
        except Exception as exc:
            print(f"[dataset] FAILED ({type(exc).__name__}: {exc}). Trying fallback.")
    if ds is None:
        raise RuntimeError(
            "Could not stream BigEarthNet-S2 from any HuggingFace repo. "
            "Check internet / HF token, or attach a BigEarthNet Kaggle Dataset."
        )

    val_stream = ds.take(CFG["val_skip"])
    train_stream = ds.skip(CFG["val_skip"])

    buf = CFG["batch_size"] * 50
    train_ds = _StreamingBuffer(train_stream, train_tf, buffer_size=buf, is_train=True)
    val_ds = _StreamingBuffer(val_stream, val_tf, buffer_size=CFG["batch_size"] * 8, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    steps = len(train_ds) // CFG["batch_size"]
    print(f"[dataset] Train steps/epoch: {steps} | Val batches capped: {CFG['val_steps_per_epoch']}")

    # Sanity check
    x, y = next(iter(train_loader))
    print(f"[dataset] Batch check: images {tuple(x.shape)} {x.dtype} | labels {tuple(y.shape)} {y.dtype}")
    return train_loader, val_loader, steps


# ===================================================================
#  Model
# ===================================================================

class MultiLabelDinoV2(nn.Module):
    """DINOv2-B backbone + 2-layer multi-label head.

    Architecture MUST match backend/app/models/dinov2.py exactly.
    Head: Linear(768→512) → ReLU → Dropout(0.3) → Linear(512→43).
    Forward returns raw logits (no sigmoid).
    """

    def __init__(self, backbone=CFG["backbone"], dim=CFG["backbone_dim"],
                 num_classes=CFG["num_classes"], dropout=CFG["head_dropout"],
                 pretrained=True):
        super().__init__()
        import timm  # noqa

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(dim, 512)),
            ("act", nn.ReLU(inplace=True)),
            ("drop", nn.Dropout(dropout)),
            ("fc2", nn.Linear(512, num_classes)),
        ]))

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n=2):
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            return
        for i in range(len(blocks) - n, len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True

    def unfreeze_full(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def _build_model():
    model = MultiLabelDinoV2(pretrained=True).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Total params: {total:,} (~{total / 1e6:.0f}M)")
    if N_GPUS > 1:
        model = nn.DataParallel(model)
        print(f"[model] Wrapped in DataParallel across {N_GPUS} GPUs.")
    return model


# ===================================================================
#  Metrics / checkpoint / curves
# ===================================================================

@torch.no_grad()
def _compute_metrics(logits, targets, threshold=CFG["threshold"]):
    """Return (f1_micro, f1_macro, hamming)."""
    from sklearn.metrics import f1_score, hamming_loss  # noqa

    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    tgt = targets.cpu().numpy().astype(int)
    return (
        f1_score(tgt, preds, average="micro", zero_division=0),
        f1_score(tgt, preds, average="macro", zero_division=0),
        hamming_loss(tgt, preds),
    )


class _EarlyStopping:
    def __init__(self, patience, min_delta, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.counter = 0
        self.triggered = False

    def step(self, value):
        improved = (value - self.best > self.min_delta) if self.mode == "max" \
            else (self.best - value > self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
            print(
                f"\n!!! Early stopping triggered "
                f"(no {self.mode}-improvement for {self.patience} epochs, "
                f"best={self.best:.4f}) !!!"
            )
            return True
        return False


def _save_checkpoint(model, epoch, optimizer, scheduler, train_m, val_m, is_best, ckpt_dir):
    raw = model.module if isinstance(model, nn.DataParallel) else model
    payload = {
        "epoch": epoch,
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_metrics": train_m,
        "val_metrics": val_m,
        "config": CFG,
    }
    torch.save(payload, ckpt_dir / f"model_epoch_{epoch}.pth")
    if is_best:
        torch.save(payload, ckpt_dir / "model_best.pth")
        print(f"   * new best (val_f1_micro={val_m['f1_micro']:.4f}) -> model_best.pth")


def _plot_curves(history, epoch, curve_dir):
    import matplotlib
    matplotlib.use("Agg")  # non-interactive — never blocks training
    import matplotlib.pyplot as plt  # noqa

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(ep, history["train_loss"], label="train")
    axes[0, 0].plot(ep, history["val_loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(ep, history["val_f1_micro"], label="F1 micro")
    axes[0, 1].plot(ep, history["val_f1_macro"], label="F1 macro")
    axes[0, 1].set_title("F1 Score")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(ep, history["val_hamming"])
    axes[1, 0].set_title("Hamming Loss (lower=better)")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].grid(True)

    axes[1, 1].plot(ep, history["lr"])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(True)

    plt.tight_layout()
    out = curve_dir / f"curves_epoch_{epoch}.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"   Saved curves -> {out}")


# ===================================================================
#  Main — three-phase training loop
# ===================================================================

def main():
    # --- Banner ---
    print("=" * 70)
    print("  Dinov2-BigEarthS2  —  Training Pipeline")
    print("=" * 70)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Device  : {DEVICE}  |  GPUs: {N_GPUS}")
    print(f"  Batch   : {CFG['batch_size']}  (effective ~{CFG['batch_size'] * CFG['grad_accum_steps']})")
    print(f"  AMP     : {CFG['use_amp']}")
    print(f"  Classes : {CFG['num_classes']}")
    print("=" * 70)

    # --- Output dirs ---
    work = Path(CFG["output_dir"])
    ckpt_dir = work / "checkpoints"
    curve_dir = work / "training_curves"
    log_dir = work / "logs"
    for d in (ckpt_dir, curve_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"  Outputs -> {work}\n")

    # --- Dataset ---
    print("[1/4] Building streaming dataset …")
    train_loader, val_loader, train_steps = _build_dataloaders()
    print()

    # --- Model + criterion ---
    print("[2/4] Building model …")
    import timm  # noqa
    model = _build_model()
    criterion = nn.BCEWithLogitsLoss()
    print()

    # --- TensorBoard + history ---
    print("[3/4] Starting training …")
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
    history = {k: [] for k in
               ("train_loss", "val_loss", "val_f1_micro", "val_f1_macro", "val_hamming", "lr", "epoch_time")}
    early = _EarlyStopping(CFG["es_patience"], CFG["es_min_delta"])
    best_f1_micro = -1.0
    global_epoch = 0

    def _unwrap(m):
        return m.module if isinstance(m, nn.DataParallel) else m

    # --- Phase loop ---
    for phase in CFG["phases"]:
        pname = phase["name"]
        n_epochs = phase["epochs"]
        lr = phase["lr"]
        unfreeze = phase["unfreeze"]

        print(f"\n{'=' * 70}")
        print(f"  PHASE: {pname}  |  epochs={n_epochs}  |  lr={lr}  |  unfreeze={unfreeze}")
        print(f"{'=' * 70}")

        # Freeze / unfreeze
        raw = _unwrap(model)
        raw.freeze_all()
        if unfreeze == "last_2_blocks":
            raw.unfreeze_last_n_blocks(2)
        elif unfreeze == "full":
            raw.unfreeze_full()
        for p in raw.classifier.parameters():
            p.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

        optimizer = torch.optim.AdamW(
            trainable, lr=lr, betas=CFG["betas"], weight_decay=CFG["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=CFG["sched_mode"], factor=CFG["sched_factor"],
            patience=CFG["sched_patience"], min_lr=CFG["sched_min_lr"],
        )
        scaler = GradScaler(enabled=CFG["use_amp"])

        for epoch in range(1, n_epochs + 1):
            global_epoch += 1
            t0 = time.time()
            model.train()

            # ---- Train ----
            running_loss = 0.0
            accum = 0
            optimizer.zero_grad(set_to_none=True)
            for step, (imgs, labels) in enumerate(train_loader, 1):
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with autocast(enabled=CFG["use_amp"]):
                    logits = model(imgs)
                    loss = criterion(logits, labels) / CFG["grad_accum_steps"]
                scaler.scale(loss).backward()
                running_loss += loss.item() * CFG["grad_accum_steps"]
                accum += 1
                if accum >= CFG["grad_accum_steps"]:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                if step % 50 == 0:
                    print(
                        f"  epoch {global_epoch} step {step}/{train_steps} "
                        f"loss={loss.item() * CFG['grad_accum_steps']:.4f}",
                        end="\r",
                    )
            train_loss = running_loss / max(1, train_steps)

            # ---- Validate ----
            model.eval()
            v_loss = 0.0
            all_logits, all_tgt = [], []
            with torch.no_grad():
                for vi, (imgs, labels) in enumerate(val_loader):
                    if vi >= CFG["val_steps_per_epoch"]:
                        break
                    imgs = imgs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    with autocast(enabled=CFG["use_amp"]):
                        logits = model(imgs)
                        v_loss += criterion(logits, labels).item()
                    all_logits.append(logits.float().cpu())
                    all_tgt.append(labels.cpu())

            val_loss = v_loss / max(1, min(CFG["val_steps_per_epoch"], len(val_loader)))
            logits_cat = torch.cat(all_logits)
            tgt_cat = torch.cat(all_tgt)
            f1m, f1M, h = _compute_metrics(logits_cat, tgt_cat)

            cur_lr = optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            # ---- Record ----
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1_micro"].append(f1m)
            history["val_f1_macro"].append(f1M)
            history["val_hamming"].append(h)
            history["lr"].append(cur_lr)
            history["epoch_time"].append(dt)

            writer.add_scalar("Loss/train", train_loss, global_epoch)
            writer.add_scalar("Loss/val", val_loss, global_epoch)
            writer.add_scalar("F1/micro", f1m, global_epoch)
            writer.add_scalar("F1/macro", f1M, global_epoch)
            writer.add_scalar("Hamming/val", h, global_epoch)
            writer.add_scalar("LR", cur_lr, global_epoch)

            print(
                f"\n  epoch {global_epoch} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"f1_micro={f1m:.4f} f1_macro={f1M:.4f} hamming={h:.4f} "
                f"lr={cur_lr:.2e} time={dt:.1f}s"
            )

            # ---- Scheduler + checkpoint ----
            scheduler.step(f1m)
            is_best = f1m > best_f1_micro + CFG["es_min_delta"]
            if is_best:
                best_f1_micro = f1m
            _save_checkpoint(
                model, global_epoch, optimizer, scheduler,
                {"loss": train_loss},
                {"loss": val_loss, "f1_micro": f1m, "f1_macro": f1M, "hamming": h},
                is_best, ckpt_dir,
            )

            if global_epoch % 5 == 0:
                _plot_curves(history, global_epoch, curve_dir)

            # ---- Early stopping ----
            if early.step(f1m):
                break

        if early.triggered:
            break

    writer.close()

    # --- Export ---
    print("\n[4/4] Exporting results …")
    with open(log_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved -> {log_dir / 'training_history.json'}")

    _plot_curves(history, global_epoch, curve_dir)

    print("\n--- Checkpoints ---")
    for p in sorted(ckpt_dir.glob("*.pth")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    print(f"\nTraining finished. Best val F1 micro: {best_f1_micro:.4f}")
    print("Download model_best.pth and place it in backend/checkpoints/")


if __name__ == "__main__":
    main()
