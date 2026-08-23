from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class SmallAndBigConfig:
    # Model
    num_layers: Optional[int] = None      # total transformer layers; None -> read from model.config
    vision_token_ids: Optional[Set[int]] = None  # ids marking visual tokens; None -> auto-detect

    # Keyframe selection (CLIP + MMR)
    alpha: float = 0.5          # keep ratio of frames before feeding the LLM
    mmr_lambda: float = 0.7     # MMR trade-off: relevance vs. diversity

    # Global visual-token selection (attention-based)
    beta: float = 0.67          # fraction of visual tokens kept after prefill

    # Layerwise progressive pruning
    # Layers 0..layer_id1-1 keep visual tokens ("big");
    # layers >= layer_id1 drop all visual tokens ("small").
    # None -> round(0.75 * num_layers): 27 for 3B(36L), 48 for 32B(64L).
    layer_id1: Optional[int] = None
    compression_ratio: float = 0.75
    alloc_steepness: float = 0.5
    min_tokens: int = 1

    # Selective fusion of dropped tokens into kept ones
    fusion_ratio: float = 0.3
    fusion_alpha: float = 0.2
    fusion_temperature: float = 1.0

    def __post_init__(self):
        if self.layer_id1 is None and self.num_layers is not None:
            self.layer_id1 = round(0.75 * self.num_layers)
