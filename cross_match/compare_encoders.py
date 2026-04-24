"""Compare DINOv2-small vs SigLIP2-base feature quality for UI element matching.

Tests:
1. Patch discriminability: How distinct are patches within a single image?
   Higher std = encoder distinguishes different UI elements better.
2. Source-Target similarity: Do independent-layout pairs look different?
   Lower cos for independent = encoder captures layout, not just texture.
3. Cross-image variance: How different do various screens look in feature space?
   Higher std = more discriminative between different screens.

Usage:
    python -m cross_match.compare_encoders --data-dir data/cross_match_v5.1_20k
    python -m cross_match.compare_encoders --data-dir data/cross_match_v5.1_20k --num-samples 20
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_dinov2(device):
    print("Loading DINOv2-small...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {params:.0f}M, dim: 384, patches: 1369 (518px)")
    tf = transforms.Compose([
        transforms.Resize((518, 518)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf, 384


def load_siglip2(device):
    print("Loading SigLIP2-base...")
    from transformers import SiglipVisionModel
    model = SiglipVisionModel.from_pretrained('google/siglip2-base-patch16-256').to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    dim = model.config.hidden_size
    print(f"  Params: {params:.0f}M, dim: {dim}, patches: 256 (256px)")
    tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return model, tf, dim


def encode_dinov2(model, tf, img_path, device):
    img = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.forward_features(img)
        return out['x_norm_patchtokens'].squeeze()


def encode_siglip2(model, tf, img_path, device):
    img = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(img).last_hidden_state.squeeze()


def patch_sim_stats(features):
    sim = F.cosine_similarity(features.unsqueeze(0), features.unsqueeze(1), dim=2)
    return sim.mean().item(), sim.std().item(), sim.min().item()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load models
    d_model, d_tf, d_dim = load_dinov2(device)
    s_model, s_tf, s_dim = load_siglip2(device)

    # Load annotations
    ann = json.load(open(os.path.join(args.data_dir, 'annotations.json')))
    pairs = ann['pairs']

    # --- Test 1: Patch Discriminability ---
    print("\n" + "=" * 60)
    print("Test 1: Patch Discriminability (within single image)")
    print("Higher std = patches more distinct = better for element matching")
    print("=" * 60 + "\n")

    d_stds, s_stds = [], []
    for p in pairs[:args.num_samples]:
        src_path = os.path.join(args.data_dir, p['source']['image'])
        if not os.path.exists(src_path):
            continue
        d_feat = encode_dinov2(d_model, d_tf, src_path, device)
        s_feat = encode_siglip2(s_model, s_tf, src_path, device)
        d_mean, d_std, d_min = patch_sim_stats(d_feat)
        s_mean, s_std, s_min = patch_sim_stats(s_feat)
        d_stds.append(d_std)
        s_stds.append(s_std)
        arch = p.get('archetype', '?')
        print(f"  {p['id'][:20]:20s} ({arch:15s}) | DINOv2: mean={d_mean:.3f} std={d_std:.3f} | SigLIP2: mean={s_mean:.3f} std={s_std:.3f}")

    avg_d_std = sum(d_stds) / len(d_stds) if d_stds else 0
    avg_s_std = sum(s_stds) / len(s_stds) if s_stds else 0
    print(f"\n  Average patch sim std: DINOv2={avg_d_std:.3f}  SigLIP2={avg_s_std:.3f}")
    print(f"  Winner: {'SigLIP2' if avg_s_std > avg_d_std else 'DINOv2'} (higher = more discriminative)")

    # --- Test 2: Source-Target Similarity by Strategy ---
    print("\n" + "=" * 60)
    print("Test 2: Source vs Target Global Feature Similarity")
    print("Independent: should be LOW (different layouts)")
    print("Proportional: should be HIGH (similar layouts)")
    print("=" * 60 + "\n")

    for strategy in ['independent', 'proportional', 'identity']:
        strat_pairs = [p for p in pairs if p.get('pair_strategy') == strategy][:args.num_samples]
        if not strat_pairs:
            continue
        d_sims, s_sims = [], []
        for p in strat_pairs:
            src_path = os.path.join(args.data_dir, p['source']['image'])
            tgt_path = os.path.join(args.data_dir, p['target']['image'])
            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                continue
            d_src = encode_dinov2(d_model, d_tf, src_path, device).mean(0, keepdim=True)
            d_tgt = encode_dinov2(d_model, d_tf, tgt_path, device).mean(0, keepdim=True)
            s_src = encode_siglip2(s_model, s_tf, src_path, device).mean(0, keepdim=True)
            s_tgt = encode_siglip2(s_model, s_tf, tgt_path, device).mean(0, keepdim=True)
            d_cos = F.cosine_similarity(d_src, d_tgt).item()
            s_cos = F.cosine_similarity(s_src, s_tgt).item()
            d_sims.append(d_cos)
            s_sims.append(s_cos)

        d_avg = sum(d_sims) / len(d_sims) if d_sims else 0
        s_avg = sum(s_sims) / len(s_sims) if s_sims else 0
        print(f"  {strategy:15s} ({len(d_sims)} pairs): DINOv2={d_avg:.3f}  SigLIP2={s_avg:.3f}")

    # --- Test 3: Cross-Image Variance ---
    print("\n" + "=" * 60)
    print("Test 3: Cross-Image Feature Variance")
    print("Higher std = more discriminative between different images")
    print("=" * 60 + "\n")

    d_globals, s_globals = [], []
    for p in pairs[:min(args.num_samples, 20)]:
        src_path = os.path.join(args.data_dir, p['source']['image'])
        if not os.path.exists(src_path):
            continue
        d_globals.append(encode_dinov2(d_model, d_tf, src_path, device).mean(0))
        s_globals.append(encode_siglip2(s_model, s_tf, src_path, device).mean(0))

    d_stack = torch.stack(d_globals)
    s_stack = torch.stack(s_globals)
    d_cross = F.cosine_similarity(d_stack.unsqueeze(0), d_stack.unsqueeze(1), dim=2)
    s_cross = F.cosine_similarity(s_stack.unsqueeze(0), s_stack.unsqueeze(1), dim=2)

    print(f"  DINOv2: cross-image cos mean={d_cross.mean():.3f} std={d_cross.std():.3f}")
    print(f"  SigLIP2: cross-image cos mean={s_cross.mean():.3f} std={s_cross.std():.3f}")
    print(f"  Winner: {'SigLIP2' if s_cross.std() > d_cross.std() else 'DINOv2'}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Patch discriminability:  DINOv2={avg_d_std:.3f}  SigLIP2={avg_s_std:.3f}  {'SigLIP2 wins' if avg_s_std > avg_d_std else 'DINOv2 wins'}")
    d_indep = sum(d_sims) / len(d_sims) if d_sims else 0
    s_indep = sum(s_sims) / len(s_sims) if s_sims else 0
    print(f"  Cross-image variance:    DINOv2={d_cross.std():.3f}  SigLIP2={s_cross.std():.3f}  {'SigLIP2 wins' if s_cross.std() > d_cross.std() else 'DINOv2 wins'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()
    main(args)
