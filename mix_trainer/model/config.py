from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ModelSize(Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class ModelConfig:
    model_size: ModelSize = ModelSize.SMALL
    vocab_size: int = 8000
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True
    weight_tying: bool = True

    PRESETS = {
        ModelSize.TINY: {"n_layer": 4, "n_head": 2, "n_embd": 64, "block_size": 128},
        ModelSize.SMALL: {"n_layer": 6, "n_head": 4, "n_embd": 256, "block_size": 256},
        ModelSize.MEDIUM: {"n_layer": 8, "n_head": 8, "n_embd": 512, "block_size": 512},
        ModelSize.LARGE: {"n_layer": 12, "n_head": 12, "n_embd": 768, "block_size": 1024},
    }

    @classmethod
    def from_preset(cls, model_size: ModelSize, vocab_size: int = 8000) -> "ModelConfig":
        preset = cls.PRESETS[model_size]
        return cls(
            model_size=model_size,
            vocab_size=vocab_size,
            n_layer=preset["n_layer"],
            n_head=preset["n_head"],
            n_embd=preset["n_embd"],
            block_size=preset["block_size"],
        )

    def get_num_params(self) -> int:
        embd_params = self.vocab_size * self.n_embd * (1 if self.weight_tying else 2)
        pos_params = self.block_size * self.n_embd
        attn_params = self.n_layer * (4 * self.n_embd * self.n_embd + 2 * self.n_embd)
        mlp_params = self.n_layer * (
            self.n_embd * 4 * self.n_embd + 4 * self.n_embd * self.n_embd + 2 * 4 * self.n_embd
        )
        ln_params = self.n_layer * 2 * 2 * self.n_embd + 2 * self.n_embd
        return embd_params + pos_params + attn_params + mlp_params + ln_params

    def to_dict(self) -> dict:
        return {
            "model_size": self.model_size.value,
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "bias": self.bias,
            "weight_tying": self.weight_tying,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        d["model_size"] = ModelSize(d["model_size"])
        return cls(**d)
