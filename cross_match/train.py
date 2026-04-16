"""Training script for CrossMatch model.

Two training phases:
  Phase 1 (frozen encoder): Train cross-attention head on precomputed features.
  Phase 2 (fine-tune): Unfreeze encoder with very low lr for final epochs.

Usage:
    python -m cross_match.train --data-dir data/cross_match --output-dir checkpoints/cross_match --device mps
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


# Reference screen sizes for pixel distance calculation
REF_SCREEN = (1170, 2532)  # iOS target


def cache_features(model: CrossMatchModel, dataset: CrossMatchDataset, cache_dir: str, device: str):
    """Precompute DINOv2 features for all images and save to disk."""
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

    seen = set()
    image_paths = []
    for sample in dataset.samples:
        for key, subdir in [("source_image_path", "source"), ("target_image_path", "target")]:
            path = sample[key]
            if path not in seen:
                seen.add(path)
                stem = os.path.splitext(os.path.basename(path))[0]
                image_paths.append((path, os.path.join(cache_dir, subdir, f"{stem}.pt")))

    print(f"  Caching features for {len(image_paths)} unique images...")

    with torch.no_grad():
        for i, (img_path, cache_path) in enumerate(image_paths):
            if os.path.exists(cache_path):
                continue
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            features = model._encode_image(tensor)
            torch.save(features.squeeze(0).cpu(), cache_path)

            if (i + 1) % 50 == 0:
                print(f"    Cached {i + 1}/{len(image_paths)}")

    print(f"  Feature caching complete.")


def compute_loss(
    outputs: dict,
    target_coords: torch.Tensor,
    source_action: torch.Tensor,
    coord_weight: float = 1.0,
    action_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Combined coordinate regression + action classification loss."""
    pred_coords = outputs["target_coords"]
    action_logits = outputs["action_logits"]

    is_scroll = (source_action == 1).float()
    coord_mask = torch.stack([
        torch.ones_like(is_scroll),
        torch.ones_like(is_scroll),
        is_scroll,
        is_scroll,
    ], dim=1)

    coord_loss_raw = nn.functional.smooth_l1_loss(pred_coords, target_coords, reduction="none")
    coord_loss = (coord_loss_raw * coord_mask).sum() / coord_mask.sum()

    action_loss = nn.functional.cross_entropy(action_logits, source_action)

    total = coord_weight * coord_loss + action_weight * action_loss

    return total, {
        "total": total.item(),
        "coord": coord_loss.item(),
        "action": action_loss.item(),
    }


def train_epoch(model, loader, optimizer, device, config, use_cached=False):
    model.train()
    if model.config.freeze_encoder:
        model.encoder.eval()

    total_loss = 0.0
    total_coord = 0.0
    total_action = 0.0
    n = 0

    for batch_idx, batch in enumerate(loader):
        if use_cached:
            src_feat = batch["source_features"].to(device)
            tgt_feat = batch["target_features"].to(device)
        else:
            src_img = batch["source_image"].to(device)
            tgt_img = batch["target_image"].to(device)

        src_coords = batch["source_coords"].to(device)
        src_action = batch["source_action"].to(device)
        tgt_coords = batch["target_coords"].to(device)

        if use_cached:
            outputs = model.forward_cached(src_feat, src_coords, src_action, tgt_feat)
        else:
            outputs = model(src_img, src_coords, src_action, tgt_img)

        loss, loss_dict = compute_loss(
            outputs, tgt_coords, src_action,
            config.coord_loss_weight, config.action_loss_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_dict["total"]
        total_coord += loss_dict["coord"]
        total_action += loss_dict["action"]
        n += 1

    return {"loss": total_loss / n, "coord_loss": total_coord / n, "action_loss": total_action / n}


@torch.no_grad()
def eval_epoch(model, loader, device, config, use_cached=False):
    model.eval()
    total_loss = 0.0
    all_dists = []
    action_correct = 0
    action_total = 0
    n = 0

    for batch in loader:
        if use_cached:
            src_feat = batch["source_features"].to(device)
            tgt_feat = batch["target_features"].to(device)
        else:
            src_img = batch["source_image"].to(device)
            tgt_img = batch["target_image"].to(device)

        src_coords = batch["source_coords"].to(device)
        src_action = batch["source_action"].to(device)
        tgt_coords = batch["target_coords"].to(device)

        if use_cached:
            outputs = model.forward_cached(src_feat, src_coords, src_action, tgt_feat)
        else:
            outputs = model(src_img, src_coords, src_action, tgt_img)

        loss, loss_dict = compute_loss(outputs, tgt_coords, src_action, config.coord_loss_weight, config.action_loss_weight)
        total_loss += loss_dict["total"]

        # Per-sample normalized L2 distance (first 2 coords = primary point)
        pred = outputs["target_coords"]
        dists = torch.sqrt(((pred[:, :2] - tgt_coords[:, :2]) ** 2).sum(dim=1))
        all_dists.extend(dists.cpu().tolist())

        # Action accuracy
        pred_action = outputs["action_logits"].argmax(dim=1)
        action_correct += (pred_action == src_action).sum().item()
        action_total += src_action.shape[0]
        n += 1

    if not all_dists:
        return {"loss": 0, "coord_dist_norm": 0, "coord_dist_px": 0, "hit_20px": 0, "hit_50px": 0, "action_acc": 0}

    # Convert normalized distances to pixel distances (using reference screen)
    diag = math.sqrt(REF_SCREEN[0] ** 2 + REF_SCREEN[1] ** 2)
    px_dists = [d * diag for d in all_dists]

    sorted_px = sorted(px_dists)
    mean_px = sum(px_dists) / len(px_dists)
    median_px = sorted_px[len(sorted_px) // 2]
    hit_20 = sum(1 for d in px_dists if d <= 20) / len(px_dists)
    hit_50 = sum(1 for d in px_dists if d <= 50) / len(px_dists)
    hit_100 = sum(1 for d in px_dists if d <= 100) / len(px_dists)

    return {
        "loss": total_loss / n,
        "coord_dist_norm": sum(all_dists) / len(all_dists),
        "coord_dist_px_mean": mean_px,
        "coord_dist_px_median": median_px,
        "hit_20px": hit_20,
        "hit_50px": hit_50,
        "hit_100px": hit_100,
        "action_acc": action_correct / action_total if action_total > 0 else 0.0,
    }


def main(args):
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        cache_features=not args.no_cache,
        finetune_epochs=args.finetune_epochs,
        log_every=args.log_every,
    )
    model_config = ModelConfig()

    os.makedirs(config.output_dir, exist_ok=True)
    device = config.device

    print(f"Initializing CrossMatch model...")
    model = CrossMatchModel(model_config).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total / 1e6:.1f}M | Trainable: {trainable / 1e6:.1f}M")

    # Dataset
    raw_dataset = CrossMatchDataset(config.data_dir, model_config.image_size)
    print(f"  Total samples (actions): {len(raw_dataset)}")

    use_cached = config.cache_features
    if use_cached:
        cache_features(model, raw_dataset, config.cache_dir, device)
        dataset = CachedFeatureDataset(config.data_dir, config.cache_dir, model_config.image_size)
    else:
        dataset = raw_dataset

    # Split 90/10
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    # Use num_workers=0 on MPS to avoid fork issues
    nw = 0 if device == "mps" else config.num_workers

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=(device != "mps"))
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=(device != "mps"))

    # Also keep raw image loaders for fine-tune phase
    if use_cached:
        raw_train, raw_val = random_split(raw_dataset, [n_train, n_val],
                                           generator=torch.Generator().manual_seed(42))
        raw_train_loader = DataLoader(raw_train, batch_size=config.batch_size, shuffle=True,
                                      num_workers=nw, pin_memory=(device != "mps"))
        raw_val_loader = DataLoader(raw_val, batch_size=config.batch_size, shuffle=False,
                                    num_workers=nw, pin_memory=(device != "mps"))

    print(f"  Train: {n_train} | Val: {n_val} | Device: {device}")
    print(f"  Epochs: {config.num_epochs} (frozen: {config.num_epochs - config.finetune_epochs}, fine-tune: {config.finetune_epochs})")
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

    # Training history
    history = []
    best_val_loss = float("inf")
    training_start = time.time()

    for epoch in range(config.num_epochs):
        t0 = time.time()
        is_finetune = epoch >= (config.num_epochs - config.finetune_epochs)
        cur_use_cached = use_cached

        # Phase 2: unfreeze encoder
        if is_finetune and model_config.freeze_encoder:
            if epoch == config.num_epochs - config.finetune_epochs:
                print("\n" + "=" * 60)
                print("PHASE 2: Fine-tuning encoder (unfrozen, low lr)")
                print("=" * 60)
                for param in model.encoder.parameters():
                    param.requires_grad = True
                optimizer.add_param_group({
                    "params": model.encoder.parameters(),
                    "lr": config.finetune_lr,
                })
            cur_use_cached = False
            cur_train_loader = raw_train_loader if use_cached else train_loader
            cur_val_loader = raw_val_loader if use_cached else val_loader
        else:
            cur_train_loader = train_loader
            cur_val_loader = val_loader

        train_m = train_epoch(model, cur_train_loader, optimizer, device, config, cur_use_cached)
        val_m = eval_epoch(model, cur_val_loader, device, config, cur_use_cached)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        phase = "FT" if is_finetune else "FR"

        entry = {
            "epoch": epoch + 1,
            "phase": phase,
            "lr": lr,
            "train_loss": train_m["loss"],
            "train_coord_loss": train_m["coord_loss"],
            "train_action_loss": train_m["action_loss"],
            "val_loss": val_m["loss"],
            "val_coord_dist_px_mean": val_m["coord_dist_px_mean"],
            "val_coord_dist_px_median": val_m["coord_dist_px_median"],
            "val_hit_20px": val_m["hit_20px"],
            "val_hit_50px": val_m["hit_50px"],
            "val_hit_100px": val_m["hit_100px"],
            "val_action_acc": val_m["action_acc"],
            "epoch_time": elapsed,
        }
        history.append(entry)

        print(f"[{phase}] Epoch {epoch+1:3d}/{config.num_epochs} ({elapsed:.1f}s) lr={lr:.2e} | "
              f"train={train_m['loss']:.4f} val={val_m['loss']:.4f} | "
              f"px_mean={val_m['coord_dist_px_mean']:.0f} px_med={val_m['coord_dist_px_median']:.0f} | "
              f"@20={val_m['hit_20px']:.0%} @50={val_m['hit_50px']:.0%} @100={val_m['hit_100px']:.0%} | "
              f"act={val_m['action_acc']:.0%}")

        # Save best
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            save_path = os.path.join(config.output_dir, "best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "val_metrics": val_m,
            }, save_path)
            print(f"  --> New best model saved (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % config.save_every_epoch == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
            }, os.path.join(config.output_dir, f"epoch_{epoch + 1}.pt"))

    total_time = time.time() - training_start

    # Save training log
    log = {
        "config": {
            "model_params_total": total,
            "model_params_trainable": trainable,
            "num_train": n_train,
            "num_val": n_val,
            "device": device,
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
        },
        "best_val_loss": best_val_loss,
        "total_training_time_seconds": total_time,
        "history": history,
    }
    log_path = os.path.join(config.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete in {total_time:.0f}s")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Training log: {log_path}")
    print(f"Best model:   {os.path.join(config.output_dir, 'best.pt')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CrossMatch model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="checkpoints/cross_match")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-cache", action="store_true", help="Don't precompute encoder features")
    args = parser.parse_args()
    main(args)
