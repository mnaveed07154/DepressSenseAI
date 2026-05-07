from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Config:
    project_root: str = "."
    metadata_csv: str = "data/metadata/metadata.csv"
    processed_dir: str = "data/processed"
    results_dir: str = "results"
    sample_rate: int = 16000
    segment_seconds: float = 5.0
    overlap_seconds: float = 2.5
    ssl_model_name: str = "facebook/wav2vec2-base"
    use_ssl: bool = True
    ssl_dim: int = 768
    acoustic_dim: int = 64
    fusion_dim: int = 256
    temporal_hidden: int = 128
    temporal_layers: int = 1
    attention_dim: int = 128
    num_classes: int = 2
    num_domains: int = 3
    dropout: float = 0.3
    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    alpha_severity: float = 0.5
    beta_domain: float = 0.1
    early_stopping_patience: int = 6
    seed: int = 42
    device: str = "auto"
    demo_mode: bool = False
    cache_features: bool = True

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Config(**data)
