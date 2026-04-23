"""Training script for CrossMatch model.

Two training phases:
  Phase 1 (frozen encoder): Train cross-attention head on precomputed features.
  Phase 2 (fine-tune): Unfreeze encoder with very low lr for final epochs.

Usage:
    # Calibration run (5K pairs, 5 epochs)
    python -m cross_match.train --data-dir data/cross_match_v5 --output-dir checkpoints/v5_calibration \
        --epochs 5 --finetune-epochs 0 --batch-size 64 --device cuda

    # Full run Phase 1 (20K pairs, 30 frozen epochs)
    python -m cross_match.train --data-dir data/cross_match_v5 --output-dir checkpoints/v5_full \
        --epochs 30 --finetune-epochs 0 --batch-size 64 --lr 1e-4 --device cuda --amp

    # Full run Phase 2 (fine-tune from Phase 1 best checkpoint)
    python -m cross_match.train --data-dir data/cross_match_v5 --output-dir checkpoints/v5_full \
        --epochs 5 --finetune-epochs 5 --batch-size 32 --lr 1e-6 --device cuda --amp \
        --resume checkpoints/v5_full/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from cross_match.config import ModelConfig, TrainConfig
from cross_match.dataset import CachedFeatureDataset, CrossMatchDataset
from cross_match.model import CrossMatchModel

REF_SCREEN = (1170, 2532)


def cache_features(model: CrossMatchModel, dataset: CrossMatchDataset, cache_dir: str, device: str, batch_size: int = 32):
    """Precompute DINOv2 features for all images in batches (10-20x faster than one-by-one)."""
    from PIL import Image
    from torchvision import transforms

    os.makedirs(os.path.join(cache_dir, "source"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "target"), exist_ok=True)

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((model.config.image_size, model.config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Collect unique image paths that need caching
    seen = set()
    to_cache = []
    for sample in dataset.samples:
        for key, subdir in [("source_image_path", "source"), ("target_image_path", "target")]:
            path = sample[key]
            if path not in seen:
                seen.add(path)
                stem = os.path.splitext(os.path.basename(path))[0]
                cache_path = os.path.join(cache_dir, subdir, "{}.pt".format(stem))
                if not os.path.exists(cache_path):
                    to_cache.append((path, cache_path))

    total = len(to_cache)
    if total == 0:
        print("  Feature cache already complete.")
        return

    print("  Caching features for {} images (batch_size={})...".format(total, batch_size))
    t0 = time.time()

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_items = to_cache[batch_start:batch_start + batch_size]
        images = []
        for img_path, _ in batch_items:
            img = Image.open(img_path).convert("RGB")
            images.append(transform(img))

        batch_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            batch_features = model._encode_image(batch_tensor)  # (B, N_patches, D)

        for j, (_, cache_path) in enumerate(batch_items):
            torch.save(batch_features[j].cpu(), cache_path)

        done = min(batch_start + batch_size, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print("    Cached {}/{} ({:.0f} img/s, ETA {:.0f}s)".format(done, total, rate, eta))

    print("  Feature caching complete in {:.0f}s".format(time.time() - t0))


def compute_loss(outputs, target_coords, source_action, coord_weight=1.0, action_weight=0.1):
    pred_coords = outputs["target_coords"]
    action_logits = outputs["action_logits"]

    is_scroll = (source_action == 1).float()
    coord_mask = torch.stack([
        torch.ones_like(is_scroll), torch.ones_like(is_scroll),
        is_scroll, is_scroll,
    ], dim=1)

    coord_loss_raw = nn.functional.smooth_l1_loss(pred_coords, target_coords, reduction="none")
    coord_loss = (coord_loss_raw * coord_mask).sum() / coord_mask.sum()
    action_loss = nn.functional.cross_entropy(action_logits, source_action)
    total = coord_weight * coord_loss + action_weight * action_loss

    return total, {"total": total.item(), "coord": coord_loss.item(), "action": action_loss.item()}


def train_epoch(model, loader, optimizer, device, config, use_cached=False, scaler=None):
    model.train()
    if model.config.freeze_encoder:
        model.encoder.eval()

    total_loss, total_coord, total_action, n = 0.0, 0.0, 0.0, 0
    use_amp = scaler is not None

    for batch in loader:
        if use_cached:
            src, tgt = batch["source_features"].to(device), batch["target_features"].to(device)
        else:
            src, tgt = batch["source_image"].to(device), batch["target_image"].to(device)

        src_coords = batch["source_coords"].to(device)
        src_action = batch["source_action"].to(device)
        tgt_coords = batch["target_coords"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model.forward_cached(src, src_coords, src_action, tgt) if use_cached else model(src, src_coords, src_action, tgt)
            loss, ld = compute_loss(outputs, tgt_coords, src_action, config.coord_loss_weight, config.action_loss_weight)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += ld["total"]; total_coord += ld["coord"]; total_action += ld["action"]; n += 1

    return {"loss": total_loss / n, "coord_loss": total_coord / n, "action_loss": total_action / n}


@torch.no_grad()
def eval_epoch(model, loader, device, config, use_cached=False, use_amp=False):
    model.eval()
    total_loss, n = 0.0, 0
    all_dists = []
    action_correct, action_total = 0, 0

    for batch in loader:
        if use_cached:
            src, tgt = batch["source_features"].to(device), batch["target_features"].to(device)
        else:
            src, tgt = batch["source_image"].to(device), batch["target_image"].to(device)

        src_coords = batch["source_coords"].to(device)
        src_action = batch["source_action"].to(device)
        tgt_coords = batch["target_coords"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model.forward_cached(src, src_coords, src_action, tgt) if use_cached else model(src, src_coords, src_action, tgt)
            loss, ld = compute_loss(outputs, tgt_coords, src_action, config.coord_loss_weight, config.action_loss_weight)
        total_loss += ld["total"]

        pred = outputs["target_coords"].float()
        dists = torch.sqrt(((pred[:, :2] - tgt_coords[:, :2]) ** 2).sum(dim=1))
        all_dists.extend(dists.cpu().tolist())

        pred_action = outputs["action_logits"].argmax(dim=1)
        action_correct += (pred_action == src_action).sum().item()
        action_total += src_action.shape[0]
        n += 1

    if not all_dists:
        return {"loss": 0, "coord_dist_norm": 0, "coord_dist_px_mean": 0, "coord_dist_px_median": 0,
                "hit_10px": 0, "hit_20px": 0, "hit_30px": 0, "hit_50px": 0, "hit_75px": 0, "hit_100px": 0,
                "action_acc": 0}

    diag = math.sqrt(REF_SCREEN[0] ** 2 + REF_SCREEN[1] ** 2)
    px_dists = [d * diag for d in all_dists]
    sorted_px = sorted(px_dists)
    nd = len(px_dists)

    return {
        "loss": total_loss / n,
        "coord_dist_norm": sum(all_dists) / nd,
        "coord_dist_px_mean": sum(px_dists) / nd,
        "coord_dist_px_median": sorted_px[nd // 2],
        "coord_dist_px_p95": sorted_px[int(nd * 0.95)] if nd > 1 else sorted_px[0],
        "hit_10px": sum(1 for d in px_dists if d <= 10) / nd,
        "hit_20px": sum(1 for d in px_dists if d <= 20) / nd,
        "hit_30px": sum(1 for d in px_dists if d <= 30) / nd,
        "hit_50px": sum(1 for d in px_dists if d <= 50) / nd,
        "hit_75px": sum(1 for d in px_dists if d <= 75) / nd,
        "hit_100px": sum(1 for d in px_dists if d <= 100) / nd,
        "hit_150px": sum(1 for d in px_dists if d <= 150) / nd,
        "action_acc": action_correct / action_total if action_total > 0 else 0.0,
    }


def write_progress(path, epoch, total_epochs, phase, train_m, val_m, elapsed, total_time):
    """Write a progress file that can be polled remotely."""
    progress = {
        "epoch": epoch, "total_epochs": total_epochs, "phase": phase,
        "train_loss": train_m["loss"], "val_loss": val_m["loss"],
        "val_px_mean": val_m["coord_dist_px_mean"], "val_px_median": val_m["coord_dist_px_median"],
        "val_hit_20px": val_m["hit_20px"], "val_hit_50px": val_m["hit_50px"],
        "val_hit_100px": val_m["hit_100px"], "val_action_acc": val_m["action_acc"],
        "epoch_time": elapsed, "total_time": total_time,
        "status": "training",
    }
    with open(path, "w") as f:
        json.dump(progress, f)


def main(args):
    config = TrainConfig(
        data_dir=args.data_dir, output_dir=args.output_dir, device=args.device,
        batch_size=args.batch_size, num_epochs=args.epochs,
        lr=args.lr, warmup_epochs=args.warmup_epochs,
        cache_features=not args.no_cache, finetune_epochs=args.finetune_epochs, log_every=args.log_every,
    )
    if args.finetune_lr:
        config.finetune_lr = args.finetune_lr
    model_config = ModelConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    device = config.device
    progress_path = os.path.join(config.output_dir, "progress.json")
    use_amp = args.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    cache_bs = args.cache_batch_size

    print("Initializing CrossMatch model...")
    model = CrossMatchModel(model_config).to(device)

    if args.resume:
        print("  Resuming from checkpoint: {}".format(args.resume))
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("  Loaded model state from epoch {}".format(ckpt.get("epoch", "?")))
        if args.finetune_epochs > 0:
            print("  Unfreezing encoder for fine-tuning")
            for param in model.encoder.parameters():
                param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("  Total params: {:.1f}M | Trainable: {:.1f}M".format(total_params / 1e6, trainable / 1e6))

    raw_dataset = CrossMatchDataset(config.data_dir, model_config.image_size)
    print("  Total samples (actions): {}".format(len(raw_dataset)))

    use_cached = config.cache_features
    if use_cached:
        cache_features(model, raw_dataset, config.cache_dir, device, batch_size=cache_bs)
        dataset = CachedFeatureDataset(config.data_dir, config.cache_dir, model_config.image_size)
    else:
        dataset = raw_dataset

    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    nw = 0 if device == "mps" else config.num_workers
    pin = device not in ("mps", "cpu")

    persistent = nw > 0
    prefetch = 4 if nw > 0 else None
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=persistent, prefetch_factor=prefetch)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=persistent, prefetch_factor=prefetch)

    if use_cached:
        raw_train, raw_val = random_split(raw_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        raw_train_loader = DataLoader(raw_train, batch_size=config.batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
        raw_val_loader = DataLoader(raw_val, batch_size=config.batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    print("  Train: {} | Val: {} | Device: {} | AMP: {}".format(n_train, n_val, device, use_amp))
    print("  Epochs: {} (frozen: {}, fine-tune: {}) | LR: {} | Batch: {}".format(
        config.num_epochs, config.num_epochs - config.finetune_epochs, config.finetune_epochs,
        config.lr, config.batch_size))
    print()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr, weight_decay=config.weight_decay,
    )

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return max(1e-2, epoch / config.warmup_epochs)
        progress = (epoch - config.warmup_epochs) / max(1, config.num_epochs - config.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    best_val_loss = float("inf")
    training_start = time.time()

    for epoch in range(config.num_epochs):
        t0 = time.time()
        is_finetune = epoch >= (config.num_epochs - config.finetune_epochs)
        cur_use_cached = use_cached

        if is_finetune and model_config.freeze_encoder:
            if epoch == config.num_epochs - config.finetune_epochs:
                print("\n" + "=" * 60)
                print("PHASE 2: Fine-tuning encoder")
                print("=" * 60)
                for param in model.encoder.parameters():
                    param.requires_grad = True
                optimizer.add_param_group({"params": model.encoder.parameters(), "lr": config.finetune_lr})
            cur_use_cached = False
            cur_train_loader = raw_train_loader if use_cached else train_loader
            cur_val_loader = raw_val_loader if use_cached else val_loader
        else:
            cur_train_loader = train_loader
            cur_val_loader = val_loader

        train_m = train_epoch(model, cur_train_loader, optimizer, device, config, cur_use_cached, scaler=scaler)
        val_m = eval_epoch(model, cur_val_loader, device, config, cur_use_cached, use_amp=use_amp)
        scheduler.step()

        elapsed = time.time() - t0
        total_time = time.time() - training_start
        lr = optimizer.param_groups[0]["lr"]
        phase = "FT" if is_finetune else "FR"

        entry = {
            "epoch": epoch + 1, "phase": phase, "lr": lr,
            "train_loss": train_m["loss"], "train_coord_loss": train_m["coord_loss"],
            "val_loss": val_m["loss"],
            "val_px_mean": val_m["coord_dist_px_mean"], "val_px_median": val_m["coord_dist_px_median"],
            "val_hit_20px": val_m["hit_20px"], "val_hit_50px": val_m["hit_50px"], "val_hit_100px": val_m["hit_100px"],
            "val_action_acc": val_m["action_acc"], "epoch_time": elapsed,
        }
        history.append(entry)

        print("[{}] Epoch {:3d}/{} ({:.1f}s) lr={:.2e} | train={:.4f} val={:.4f} | "
              "px_mean={:.0f} px_med={:.0f} p95={:.0f} | @10={:.0%} @20={:.0%} @50={:.0%} @100={:.0%} | act={:.0%}".format(
            phase, epoch + 1, config.num_epochs, elapsed, lr,
            train_m["loss"], val_m["loss"],
            val_m["coord_dist_px_mean"], val_m["coord_dist_px_median"], val_m.get("coord_dist_px_p95", 0),
            val_m.get("hit_10px", 0), val_m["hit_20px"], val_m["hit_50px"], val_m["hit_100px"], val_m["action_acc"]))

        write_progress(progress_path, epoch + 1, config.num_epochs, phase, train_m, val_m, elapsed, total_time)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                         "model_config": model_config, "val_metrics": val_m},
                        os.path.join(config.output_dir, "best.pt"))
            print("  --> New best (val_loss={:.4f})".format(best_val_loss))

        if (epoch + 1) % config.save_every_epoch == 0:
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "model_config": model_config},
                        os.path.join(config.output_dir, "epoch_{}.pt".format(epoch + 1)))

    total_time = time.time() - training_start

    log = {
        "config": {"model_params_total": total_params, "model_params_trainable": trainable,
                    "num_train": n_train, "num_val": n_val, "device": device,
                    "epochs": config.num_epochs, "batch_size": config.batch_size, "lr": config.lr},
        "best_val_loss": best_val_loss, "total_training_time_seconds": total_time, "history": history,
    }
    with open(os.path.join(config.output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # Mark progress as complete
    with open(progress_path, "w") as f:
        json.dump({"status": "complete", "epoch": config.num_epochs, "total_epochs": config.num_epochs,
                    "best_val_loss": best_val_loss, "total_time": total_time}, f)

    print("\n" + "=" * 60)
    print("Training complete in {:.0f}s ({:.1f} min)".format(total_time, total_time / 60))
    print("Best val_loss: {:.4f}".format(best_val_loss))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CrossMatch model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="checkpoints/cross_match")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--finetune-lr", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-batch-size", type=int, default=16, help="Batch size for feature caching (16 safest for CUDA compat)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (fp16)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args)
