import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Vision encoder
    encoder_name: str = "dinov2_vits14"  # DINOv2-small via torch.hub
    encoder_dim: int = 384               # DINOv2-small output dim
    freeze_encoder: bool = True
    image_size: int = 518               # DINOv2 native: 518 = 37 patches * 14

    # Cross-attention transformer
    num_layers: int = 4
    num_heads: int = 6
    hidden_dim: int = 384
    ffn_dim: int = 1536                  # 4x hidden
    dropout: float = 0.1

    # Coordinate encoder
    coord_embed_dim: int = 384
    num_freq_bands: int = 64             # Sinusoidal PE bands

    # Action types
    action_types: list = field(default_factory=lambda: ["click", "scroll"])
    num_actions: int = 2
    max_coords: int = 4                  # click: 2 (at_x, at_y), scroll: 4 (from_x, from_y, to_x, to_y)


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/cross_match"
    image_size: int = 518

    # Training
    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 5

    # Loss weights
    coord_loss_weight: float = 1.0
    action_loss_weight: float = 0.1

    # Fine-tuning (last N epochs, unfreeze encoder with low lr)
    finetune_epochs: int = 5
    finetune_lr: float = 1e-6

    # Infra
    device: str = "cuda"
    num_workers: int = 4
    output_dir: str = "checkpoints/cross_match"
    log_every: int = 50
    save_every_epoch: int = 5

    # Feature caching (precompute encoder features for faster training)
    cache_features: bool = True
    cache_dir: str = ""  # Derived from data_dir if empty

    def __post_init__(self):
        if not self.cache_dir:
            self.cache_dir = os.path.join(self.data_dir, "feature_cache")
