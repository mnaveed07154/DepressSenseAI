from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .preprocess import preprocess_file
from .feature_extraction import extract_acoustic_features, SSLEmbeddingExtractor

REQUIRED_COLUMNS = ["session_id", "subject_id", "audio_path", "label", "domain"]


def create_demo_metadata(out_csv="data/metadata/metadata.csv", n_sessions=18, sr=16000):
    import soundfile as sf
    rng = np.random.default_rng(42)
    root = Path(out_csv).parents[1] / "raw" / "demo"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_sessions):
        label = i % 2
        domain = i % 3
        dur = 6 + (i % 4)
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        freq = 120 + 30 * label + 10 * domain
        y = 0.05 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.normal(size=len(t))
        path = root / f"demo_{i:03d}.wav"
        sf.write(path, y.astype(np.float32), sr)
        rows.append({
            "session_id": f"demo_{i:03d}", "subject_id": f"subj_{i:03d}",
            "audio_path": str(path), "label": label, "severity": float(8 + 8 * label + rng.normal()) if domain == 2 else np.nan,
            "domain": domain, "dataset": ["ADC", "EATD", "PDCH"][domain]
        })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def validate_metadata(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {missing}")
    if "severity" not in df.columns:
        df["severity"] = np.nan
    if "dataset" not in df.columns:
        df["dataset"] = df["domain"].astype(str)
    return df


def subject_split(df, seed=42, test_size=0.2, val_size=0.2):
    subjects = df["subject_id"].astype(str).unique()
    train_subj, test_subj = train_test_split(subjects, test_size=test_size, random_state=seed)
    train_subj, val_subj = train_test_split(train_subj, test_size=val_size, random_state=seed)
    def assign(s):
        if s in set(train_subj): return "train"
        if s in set(val_subj): return "val"
        return "test"
    df = df.copy()
    df["split"] = df["subject_id"].astype(str).map(assign)
    return df


class DepressionSpeechDataset(Dataset):
    def __init__(self, df, cfg, split="train", ssl_extractor=None):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.cfg = cfg
        self.ssl_extractor = ssl_extractor
        self.processed_dir = Path(cfg.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _feature_cache_path(self, session_id):
        return self.processed_dir / f"{session_id}_features.npz"

    def _extract_session_features(self, row):
        session_id = str(row.session_id)
        cache = self._feature_cache_path(session_id)
        if self.cfg.cache_features and cache.exists():
            data = np.load(cache)
            return data["ssl"], data["acoustic"]
        segments = preprocess_file(row.audio_path, sr=self.cfg.sample_rate,
                                   segment_seconds=self.cfg.segment_seconds,
                                   overlap_seconds=self.cfg.overlap_seconds)
        if not segments:
            segments = [np.zeros(int(self.cfg.sample_rate * self.cfg.segment_seconds), dtype=np.float32)]
        ssl_feats, acoustic_feats = [], []
        for seg in segments:
            if self.ssl_extractor is None:
                from .feature_extraction import SSLEmbeddingExtractor
                self.ssl_extractor = SSLEmbeddingExtractor(enabled=False)
            ssl_feats.append(self.ssl_extractor.extract(seg, self.cfg.sample_rate))
            acoustic_feats.append(extract_acoustic_features(seg, self.cfg.sample_rate))
        ssl_feats = np.stack(ssl_feats).astype(np.float32)
        acoustic_feats = np.stack(acoustic_feats).astype(np.float32)
        if self.cfg.cache_features:
            np.savez_compressed(cache, ssl=ssl_feats, acoustic=acoustic_feats)
        return ssl_feats, acoustic_feats

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ssl, acoustic = self._extract_session_features(row)
        severity = row.severity if not pd.isna(row.severity) else -1.0
        has_sev = 0.0 if pd.isna(row.severity) else 1.0
        return {
            "ssl": torch.tensor(ssl, dtype=torch.float32),
            "acoustic": torch.tensor(acoustic, dtype=torch.float32),
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "severity": torch.tensor(float(severity), dtype=torch.float32),
            "has_severity": torch.tensor(float(has_sev), dtype=torch.float32),
            "domain": torch.tensor(int(row.domain), dtype=torch.long),
            "session_id": str(row.session_id)
        }


def collate_sessions(batch):
    max_len = max(item["ssl"].shape[0] for item in batch)
    ssl_dim = batch[0]["ssl"].shape[1]
    ac_dim = batch[0]["acoustic"].shape[1]
    bsz = len(batch)
    ssl = torch.zeros(bsz, max_len, ssl_dim)
    acoustic = torch.zeros(bsz, max_len, ac_dim)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["ssl"].shape[0]
        ssl[i, :n] = item["ssl"]
        acoustic[i, :n] = item["acoustic"]
        mask[i, :n] = True
    return {
        "ssl": ssl, "acoustic": acoustic, "mask": mask,
        "label": torch.stack([x["label"] for x in batch]),
        "severity": torch.stack([x["severity"] for x in batch]),
        "has_severity": torch.stack([x["has_severity"] for x in batch]),
        "domain": torch.stack([x["domain"] for x in batch]),
        "session_id": [x["session_id"] for x in batch]
    }
