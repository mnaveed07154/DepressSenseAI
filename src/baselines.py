from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from .config import Config
from .dataset import validate_metadata, subject_split, DepressionSpeechDataset
from .feature_extraction import SSLEmbeddingExtractor


def session_acoustic_matrix(ds):
    X, y = [], []
    for i in range(len(ds)):
        item = ds[i]
        X.append(item["acoustic"].mean(dim=0).numpy())
        y.append(int(item["label"]))
    return np.asarray(X), np.asarray(y)


def run_baselines(config_path=None):
    cfg = Config.load(config_path) if config_path else Config()
    df = validate_metadata(cfg.metadata_csv)
    if "split" not in df.columns:
        df = subject_split(df, seed=cfg.seed)
    ssl = SSLEmbeddingExtractor(enabled=False)
    train_ds = DepressionSpeechDataset(df, cfg, "train", ssl)
    test_ds = DepressionSpeechDataset(df, cfg, "test", ssl)
    X_train, y_train = session_acoustic_matrix(train_ds)
    X_test, y_test = session_acoustic_matrix(test_ds)
    models = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "svm_rbf": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced"))]),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=cfg.seed, class_weight="balanced")
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else pred
        rows.append({"model": name, "accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred, zero_division=0),
                     "uar": balanced_accuracy_score(y_test, pred), "auc": roc_auc_score(y_test, prob) if len(set(y_test))>1 else np.nan})
    out = Path(cfg.results_dir)/"metrics"/"baseline_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows))
    return rows

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default=None); a = p.parse_args(); run_baselines(a.config)
