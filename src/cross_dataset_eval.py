from pathlib import Path
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .config import Config
from .utils import get_device, set_seed, save_json
from .dataset import validate_metadata, DepressionSpeechDataset, collate_sessions
from .feature_extraction import SSLEmbeddingExtractor
from .models.vox_depress_net import VoxDepressNet
from .evaluate import evaluate_model


def run_cross_dataset(config_path=None, checkpoint=None, target_domain=None):
    cfg = Config.load(config_path) if config_path else Config()
    set_seed(cfg.seed); device = get_device(cfg.device)
    df = validate_metadata(cfg.metadata_csv)
    if target_domain is not None:
        df = df.copy(); df["split"] = ["test" if int(d)==int(target_domain) else "train" for d in df["domain"]]
    elif "split" not in df.columns:
        raise ValueError("metadata.csv must contain split column or pass --target_domain")
    ssl_extractor = SSLEmbeddingExtractor(cfg.ssl_model_name, device=device, enabled=cfg.use_ssl)
    test_ds = DepressionSpeechDataset(df, cfg, split="test", ssl_extractor=ssl_extractor)
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_sessions)
    model = VoxDepressNet(cfg.ssl_dim, cfg.acoustic_dim, cfg.fusion_dim, cfg.temporal_hidden,
                          cfg.temporal_layers, cfg.attention_dim, cfg.num_classes, cfg.num_domains, cfg.dropout).to(device)
    checkpoint = checkpoint or str(Path(cfg.results_dir)/"checkpoints"/"best_voxdepressnet.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
    metrics = evaluate_model(model, loader, device)
    out = Path(cfg.results_dir)/"metrics"/f"cross_dataset_domain_{target_domain if target_domain is not None else 'test'}.json"
    save_json(metrics, out)
    print(metrics)
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default=None); p.add_argument("--checkpoint", default=None); p.add_argument("--target_domain", type=int, default=None)
    a = p.parse_args(); run_cross_dataset(a.config, a.checkpoint, a.target_domain)
