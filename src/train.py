from pathlib import Path
import argparse, json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import Config
from .utils import set_seed, get_device, ensure_dir, save_json
from .dataset import create_demo_metadata, validate_metadata, subject_split, DepressionSpeechDataset, collate_sessions
from .feature_extraction import SSLEmbeddingExtractor
from .models.vox_depress_net import VoxDepressNet
from .evaluate import evaluate_model, save_confusion_matrix


def build_loaders(cfg, device):
    if cfg.demo_mode and not Path(cfg.metadata_csv).exists():
        create_demo_metadata(cfg.metadata_csv)
    df = validate_metadata(cfg.metadata_csv)
    if "split" not in df.columns:
        df = subject_split(df, seed=cfg.seed)
        Path(cfg.metadata_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.metadata_csv, index=False)
    ssl_extractor = SSLEmbeddingExtractor(cfg.ssl_model_name, device=device, enabled=cfg.use_ssl)
    loaders = {}
    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        ds = DepressionSpeechDataset(df, cfg, split, ssl_extractor=ssl_extractor)
        loaders[split] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate_sessions)
    return loaders, df


def train_one_epoch(model, loader, optimizer, device, cfg, epoch):
    model.train()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss(reduction="none")
    total = 0.0
    grl_lambda = min(1.0, epoch / max(cfg.epochs // 2, 1))
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        ssl = batch["ssl"].to(device); acoustic = batch["acoustic"].to(device); mask = batch["mask"].to(device)
        labels = batch["label"].to(device); domains = batch["domain"].to(device)
        severity = batch["severity"].to(device); has_sev = batch["has_severity"].to(device)
        out = model(ssl, acoustic, mask, grl_lambda=grl_lambda)
        cls_loss = ce(out["class_logits"], labels)
        domain_loss = ce(out["domain_logits"], domains)
        sev_raw = mse(out["severity"], severity)
        sev_loss = (sev_raw * has_sev).sum() / has_sev.sum().clamp(min=1.0)
        loss = cls_loss + cfg.alpha_severity * sev_loss + cfg.beta_domain * domain_loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def main(config_path=None):
    cfg = Config.load(config_path) if config_path else Config()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ensure_dir(cfg.results_dir)
    loaders, df = build_loaders(cfg, device)
    model = VoxDepressNet(cfg.ssl_dim, cfg.acoustic_dim, cfg.fusion_dim, cfg.temporal_hidden,
                          cfg.temporal_layers, cfg.attention_dim, cfg.num_classes, cfg.num_domains,
                          cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    best_uar, bad_epochs = -1, 0
    log = []
    ckpt = Path(cfg.results_dir) / "checkpoints" / "best_voxdepressnet.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device, cfg, epoch)
        val_metrics = evaluate_model(model, loaders["val"], device)
        scheduler.step(val_metrics.get("uar", 0))
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        log.append({k: v for k, v in row.items() if k != "confusion_matrix"})
        if val_metrics.get("uar", 0) > best_uar:
            best_uar = val_metrics.get("uar", 0); bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, ckpt)
        else:
            bad_epochs += 1
        if bad_epochs >= cfg.early_stopping_patience:
            break
    pd.DataFrame(log).to_csv(Path(cfg.results_dir) / "metrics" / "training_log.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    test_metrics = evaluate_model(model, loaders["test"], device, save_attention_path=Path(cfg.results_dir)/"metrics"/"attention_weights.csv")
    save_json(test_metrics, Path(cfg.results_dir) / "metrics" / "test_metrics.json")
    save_confusion_matrix(test_metrics["confusion_matrix"], Path(cfg.results_dir) / "plots" / "confusion_matrix.png")
    print(json.dumps(test_metrics, indent=2))
    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
