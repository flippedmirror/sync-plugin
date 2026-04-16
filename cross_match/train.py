"""Training script for CrossMatch model.

Two training phases:
  Phase 1 (frozen encoder): Train cross-attention head on precomputed features.
  Phase 2 (fine-tune): Unfreeze encoder with very low lr for final epochs.

Usage:
    # Phase 1: Train with cached features (fast)
    python -m cross_match.train --data-dir data/cross_match --output-dir checkpoints/cross_match

    # Skip feature caching (slower but simpler)
    python -m cross_match.train --data-dir data/cross_match --no-cache
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from cross_match.config import ModelConfig, TrainConfig
from cross_match.dataset import CachedFeatureDataset, CrossMatchDataset
from cross_match.model import CrossMatchModel


def cache_features(model: CrossMatchModel, dataset: CrossMatchDataset, cache_dir: str, device: str):
    """Precompute DINOv2 features for all images and save to disk."""
    from torchvision import transforms
    from PIL import Image

    os.makedirs(os.path.join(cache_dir, "source"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "target"), exist_ok=True)

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((model.config.image_size, model.config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Collect unique image paths
    seen = set()
    image_paths = []
    for sample in dataset.samples:
        for key, subdir in [("source_image_path", "source"), ("target_image_path", "target")]:
            path = sample[key]
            if path not in seen:
                seen.add(path)
                stem = os.path.splitext(os.path.basename(path))[0]
                image_paths.append((path, os.path.join(cache_dir, subdir, f"{stem}.pt")))

    print(f"Caching features for {len(image_paths)} unique images...")

    with torch.no_grad():
        for i, (img_path, cache_path) in enumerate(image_paths):
            if os.path.exists(cache_path):
                continue
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            features = model._encode_image(tensor)  # (1, N, D)
            torch.save(features.squeeze(0).cpu(), cache_path)

            if (i + 1) % 100 == 0:
                print(f"  Cached {i + 1}/{len(image_paths)}")

    print("Feature caching complete.")


def compute_loss(
    outputs: dict,
    target_coords: torch.Tensor,
    target_action: torch.Tensor,
    source_action: torch.Tensor,
    coord_weight: float = 1.0,
    action_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute combined coordinate regression + action classification loss.

    For click actions, only the first 2 coords (at_x, at_y) contribute to loss.
    For scroll actions, all 4 coords (from_x, from_y, to_x, to_y) contribute.
    """
    pred_coords = outputs["target_coords"]   # (B, 4)
    action_logits = outputs["action_logits"]  # (B, num_actions)

    # Coordinate loss: smooth L1, masked by action type
    # Build per-sample mask: click uses coords [0,1], scroll uses [0,1,2,3]
    is_click = (source_action == 0).float()   # (B,)
    is_scroll = (source_action == 1).float()

    # Mask: click -> [1, 1, 0, 0], scroll -> [1, 1, 1, 1]
    coord_mask = torch.stack([
        torch.ones_like(is_click),             # coord 0 always used
        torch.ones_like(is_click),             # coord 1 always used
        is_scroll,                              # coord 2 only for scroll
        is_scroll,                              # coord 3 only for scroll
    ], dim=1)  # (B, 4)

    coord_loss_raw = nn.functional.smooth_l1_loss(pred_coords, target_coords, reduction="none")  # (B, 4)
    coord_loss = (coord_loss_raw * coord_mask).sum() / coord_mask.sum()

    # Action classification loss
    action_loss = nn.functional.cross_entropy(action_logits, source_action)

    total = coord_weight * coord_loss + action_weight * action_loss

    return total, {
        "total": total.item(),
        "coord": coord_loss.item(),
        "action": action_loss.item(),
    }


def train_epoch(
    model: CrossMatchModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    config: TrainConfig,
    use_cached: bool = False,
) -> dict:
    model.train()
    # Keep encoder frozen during phase 1
    if config.cache_features or model.config.freeze_encoder:
        model.encoder.eval()

    total_loss = 0.0
    total_coord_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0

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
            outputs, tgt_coords, tgt_coords, src_action,
            config.coord_loss_weight, config.action_loss_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_dict["total"]
        total_coord_loss += loss_dict["coord"]
        total_action_loss += loss_dict["action"]
        num_batches += 1

        if (batch_idx + 1) % config.log_every == 0:
            print(f"    batch {batch_idx + 1}: loss={loss_dict['total']:.4f} "
                  f"coord={loss_dict['coord']:.4f} action={loss_dict['action']:.4f}")

    return {
        "loss": total_loss / num_batches,
        "coord_loss": total_coord_loss / num_batches,
        "action_loss": total_action_loss / num_batches,
    }


@torch.no_grad()
def eval_epoch(
    model: CrossMatchModel,
    loader: DataLoader,
    device: str,
    config: TrainConfig,
    use_cached: bool = False,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_coord_dist = 0.0
    action_correct = 0
    action_total = 0
    num_batches = 0

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

        loss, loss_dict = compute_loss(
            outputs, tgt_coords, tgt_coords, src_action,
            config.coord_loss_weight, config.action_loss_weight,
        )
        total_loss += loss_dict["total"]

        # Coordinate distance (L2, normalized)
        pred = outputs["target_coords"]
        dist = torch.sqrt(((pred[:, :2] - tgt_coords[:, :2]) ** 2).sum(dim=1)).mean()
        total_coord_dist += dist.item()

        # Action accuracy
        pred_action = outputs["action_logits"].argmax(dim=1)
        action_correct += (pred_action == src_action).sum().item()
        action_total += src_action.shape[0]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "coord_dist": total_coord_dist / num_batches,
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
    if config.cache_features:
        # First cache features using full dataset for image paths
        raw_dataset = CrossMatchDataset(config.data_dir, model_config.image_size)
        cache_features(model, raw_dataset, config.cache_dir, device)

        dataset = CachedFeatureDataset(config.data_dir, config.cache_dir, model_config.image_size)
        use_cached = True
    else:
        dataset = CrossMatchDataset(config.data_dir, model_config.image_size)
        use_cached = False

    # Split 90/10
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    print(f"  Train: {n_train} samples | Val: {n_val} samples")

    # Optimizer: only trainable params (encoder is frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Cosine LR schedule with warmup
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / (config.num_epochs - config.warmup_epochs)
        return 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(config.num_epochs):
        t0 = time.time()
        is_finetune = epoch >= (config.num_epochs - config.finetune_epochs)

        # Phase 2: unfreeze encoder
        if is_finetune and model_config.freeze_encoder:
            if epoch == config.num_epochs - config.finetune_epochs:
                print("\n--- Entering fine-tune phase: unfreezing encoder ---")
                for param in model.encoder.parameters():
                    param.requires_grad = True
                # Add encoder params with low lr
                optimizer.add_param_group({
                    "params": model.encoder.parameters(),
                    "lr": config.finetune_lr,
                })
            use_cached = False  # Must use raw images when encoder is unfrozen

        train_metrics = train_epoch(model, train_loader, optimizer, device, config, use_cached)
        val_metrics = eval_epoch(model, val_loader, device, config, use_cached)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{config.num_epochs} ({elapsed:.1f}s) lr={lr:.2e} | "
              f"train_loss={train_metrics['loss']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_coord_dist={val_metrics['coord_dist']:.4f} "
              f"val_action_acc={val_metrics['action_acc']:.1%}")

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = os.path.join(config.output_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "val_metrics": val_metrics,
            }, save_path)
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")

        # Periodic save
        if (epoch + 1) % config.save_every_epoch == 0:
            save_path = os.path.join(config.output_dir, f"epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
            }, save_path)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CrossMatch model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="checkpoints/cross_match")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--no-cache", action="store_true", help="Don't precompute encoder features")
    args = parser.parse_args()
    main(args)
