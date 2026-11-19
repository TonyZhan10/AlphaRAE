"""RAEv2 class-generation configuration subset."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CFGConfig:
    """CFG configuration for test-time guidance."""
    scale: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

@dataclass
class IGConfig:
    """IG configuration for test-time guidance."""
    scale: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    unconditional_scale: Optional[float] = None

@dataclass
class GuidanceConfig:
    """Guidance configuration for test-time guidance."""
    cfg: Optional[CFGConfig] = field(default_factory=CFGConfig)
    ig: Optional[IGConfig] = field(default_factory=IGConfig)

    @property
    def use_cfg(self):
        return self.cfg is not None and self.cfg.scale > 1.0

    @property
    def use_ig(self):
        return self.ig is not None and (
            self.ig.scale > 1.0
            or (self.ig.unconditional_scale is not None and self.ig.unconditional_scale != 1.0)
        )

    @property
    def any_guidance_active(self):
        return self.use_cfg or self.use_ig

@dataclass
class ConditioningArchConfig:
    """In-context conditioning architecture configuration."""
    num_t_tokens: int = 4
    num_c_tokens: int = 8
    num_cfg_omega_tokens: int = 4
    n_action_tokens: int = 4  # nwm only: action embedding tokens (rel_time adds 1 more)
