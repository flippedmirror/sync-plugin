"""CrossMatch model: frozen DINOv2 encoder + cross-attention transformer + coordinate/action heads.

Input:
  - source_image: screenshot from device A
  - source_coords: normalized action coordinates on device A
      click: (at_x, at_y, 0, 0)
      scroll: (from_x, from_y, to_x, to_y)
  - source_action: action type index (0=click, 1=scroll)
  - target_image: screenshot from device B

Output:
  - target_coords: predicted normalized coordinates on device B (4 values)
  - action_logits: predicted action type logits
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn

from cross_match.config import ModelConfig


class SinusoidalCoordEncoder(nn.Module):
    """Encode (x, y) coordinates using sinusoidal positional encoding + MLP projection."""

    def __init__(self, num_freq_bands: int = 64, output_dim: int = 384):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        # 4 input coords * 2 (sin, cos) * num_freq_bands
        input_dim = 4 * 2 * num_freq_bands
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        # Precompute frequency bands (log-spaced)
        freqs = torch.logspace(0, math.log10(512), num_freq_bands)
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 4) normalized coordinates [0, 1]
        Returns:
            (B, output_dim) coordinate embedding
        """
        # coords: (B, 4) -> (B, 4, 1) * (1, 1, F) -> (B, 4, F)
        scaled = coords.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0) * math.pi
        # (B, 4, F) -> sin/cos -> (B, 4, 2F) -> flatten -> (B, 4*2F)
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        encoded = encoded.flatten(1)  # (B, 4 * 2 * num_freq_bands)
        return self.proj(encoded)


class ActionEncoder(nn.Module):
    """Learnable embedding for action types."""

    def __init__(self, num_actions: int = 2, embed_dim: int = 384):
        super().__init__()
        self.embed = nn.Embedding(num_actions, embed_dim)

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_ids: (B,) action type indices
        Returns:
            (B, embed_dim)
        """
        return self.embed(action_ids)


class CrossAttentionBlock(nn.Module):
    """Single cross-attention + FFN block."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, Nq, D) — source context queries
            key_value: (B, Nkv, D) — target features
        Returns:
            (B, Nq, D) — updated queries
        """
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ffn(query))
        return query


class CrossMatchModel(nn.Module):
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # Frozen DINOv2 vision encoder (via timm for Python 3.9 compat)
        try:
            self.encoder = torch.hub.load("facebookresearch/dinov2", config.encoder_name)
        except TypeError:
            # Fallback: load DINOv2 via HuggingFace transformers
            from transformers import Dinov2Model
            hf_name = "facebook/dinov2-small" if "vits" in config.encoder_name else "facebook/dinov2-base"
            self.encoder = Dinov2Model.from_pretrained(hf_name)
            self._hf_encoder = True

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # Coordinate encoder (sinusoidal PE + MLP)
        self.coord_encoder = SinusoidalCoordEncoder(
            num_freq_bands=config.num_freq_bands,
            output_dim=config.hidden_dim,
        )

        # Action type encoder
        self.action_encoder = ActionEncoder(
            num_actions=config.num_actions,
            embed_dim=config.hidden_dim,
        )

        # Query construction: project concatenated [coord_embed, action_embed, source_patch_features]
        # into a small set of query tokens
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Cross-attention transformer layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                config.hidden_dim, config.num_heads, config.ffn_dim, config.dropout
            )
            for _ in range(config.num_layers)
        ])

        # Self-attention over queries (to let coord/action/spatial tokens interact)
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ffn_dim,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Output heads
        # Pool queries -> coordinate regression
        self.coord_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.max_coords),
            nn.Sigmoid(),  # Output in [0, 1] (normalized coordinates)
        )

        # Action classification head
        self.action_head = nn.Linear(config.hidden_dim, config.num_actions)

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Run frozen DINOv2 encoder. Returns patch features.

        Args:
            image: (B, 3, H, W) normalized image
        Returns:
            (B, N_patches, D) patch features
        """
        ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with ctx:
            if getattr(self, "_hf_encoder", False):
                # HuggingFace Dinov2Model
                outputs = self.encoder(image)
                patch_tokens = outputs.last_hidden_state[:, 1:]  # skip CLS token
            else:
                # torch.hub DINOv2
                features = self.encoder.forward_features(image)
                patch_tokens = features["x_norm_patchtokens"]
        return patch_tokens

    def _get_source_context(
        self,
        source_features: torch.Tensor,
        coord_embed: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Build query tokens from source context.

        Combines:
          - coordinate embedding (what location)
          - action embedding (what type of action)
          - local source patch features around the click/scroll location

        Returns (B, N_query, D) query tokens.
        """
        B = source_features.shape[0]

        # Token 1: coordinate embedding
        coord_token = coord_embed.unsqueeze(1)  # (B, 1, D)

        # Token 2: action embedding
        action_token = action_embed.unsqueeze(1)  # (B, 1, D)

        # Token 3: mean-pooled source features (global context)
        global_token = source_features.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Concatenate query tokens
        queries = torch.cat([coord_token, action_token, global_token], dim=1)  # (B, 3, D)
        queries = self.query_proj(queries)

        return queries

    def forward(
        self,
        source_image: torch.Tensor,
        source_coords: torch.Tensor,
        source_action: torch.Tensor,
        target_image: torch.Tensor,
    ) -> dict:
        """
        Args:
            source_image: (B, 3, H, W)
            source_coords: (B, 4) normalized coords — click: (x, y, 0, 0), scroll: (fx, fy, tx, ty)
            source_action: (B,) action type indices (0=click, 1=scroll)
            target_image: (B, 3, H, W)

        Returns:
            dict with:
                target_coords: (B, 4) predicted normalized coordinates
                action_logits: (B, num_actions) action type logits
        """
        # 1. Encode images
        source_features = self._encode_image(source_image)  # (B, N, D)
        target_features = self._encode_image(target_image)  # (B, N, D)

        # 2. Encode coordinates and action
        coord_embed = self.coord_encoder(source_coords)     # (B, D)
        action_embed = self.action_encoder(source_action)    # (B, D)

        # 3. Build source context queries
        queries = self._get_source_context(source_features, coord_embed, action_embed)  # (B, 3, D)

        # 4. Self-attention over queries
        queries = self.self_attn(queries)

        # 5. Cross-attend from queries to target features
        for layer in self.cross_attn_layers:
            queries = layer(queries, target_features)

        # 6. Pool queries and predict
        pooled = queries.mean(dim=1)  # (B, D)

        target_coords = self.coord_head(pooled)       # (B, 4)
        action_logits = self.action_head(pooled)       # (B, num_actions)

        return {
            "target_coords": target_coords,
            "action_logits": action_logits,
        }

    def forward_cached(
        self,
        source_features: torch.Tensor,
        source_coords: torch.Tensor,
        source_action: torch.Tensor,
        target_features: torch.Tensor,
    ) -> dict:
        """Forward pass using precomputed encoder features (for faster training)."""
        coord_embed = self.coord_encoder(source_coords)
        action_embed = self.action_encoder(source_action)

        queries = self._get_source_context(source_features, coord_embed, action_embed)
        queries = self.self_attn(queries)

        for layer in self.cross_attn_layers:
            queries = layer(queries, target_features)

        pooled = queries.mean(dim=1)

        return {
            "target_coords": self.coord_head(pooled),
            "action_logits": self.action_head(pooled),
        }
