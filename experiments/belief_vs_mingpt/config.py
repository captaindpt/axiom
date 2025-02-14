"""Configuration for belief-enhanced GPT comparison experiments."""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    # Common parameters
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024
    
    # Belief system parameters
    belief_dimension: int = 10
    belief_learning_rate: float = 0.1
    belief_temporal_decay_rate: float = 0.1
    belief_activation_threshold: float = 0.1
    belief_dominance_threshold: float = 0.5

@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 6e-4
    max_iters: int = 100000
    lr_decay: bool = True
    warmup_iters: int = 2000
    grad_norm_clip: float = 1.0
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    device: str = "cuda"  # or "cpu"

@dataclass
class EvaluationConfig:
    batch_size: int = 32
    num_samples: int = 1000
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None

@dataclass
class ExperimentConfig:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            evaluation=EvaluationConfig(**config_dict["evaluation"])
        )
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Default configuration
default_config = ExperimentConfig() 