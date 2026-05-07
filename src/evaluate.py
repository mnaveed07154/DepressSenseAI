from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, balanced_accuracy_score
from scipy.stats import pearsonr


def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)


def classification_metrics(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "uar": balanced_accuracy_score(y_true, y_pred),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        metrics["auc"] = np.nan
    return metrics


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "pearson": np.nan, "ccc": np.nan}
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pear = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "pearson": pear, "ccc": ccc}


@torch.no_grad()
def evaluate_model(model, loader, device, grl_lambda=0.0, save_attention_path=None):
    model.eval()
    y_true, y_prob, domains, domain_prob = [], [], [], []
    sev_true, sev_pred = [], []
    attn_rows = []
    for batch in loader:
        ssl = batch["ssl"].to(device); acoustic = batch["acoustic"].to(device); mask = batch["mask"].to(device)
        out = model(ssl, acoustic, mask, grl_lambda=grl_lambda)
        probs = torch.softmax(out["class_logits"], dim=1).cpu().numpy()
        dprobs = torch.softmax(out["domain_logits"], dim=1).cpu().numpy()
        y_prob.append(probs); y_true.extend(batch["label"].numpy().tolist())
        domains.extend(batch["domain"].numpy().tolist()); domain_prob.append(dprobs)
        has = batch["has_severity"].numpy() > 0
        if has.any():
            sev_true.extend(batch["severity"].numpy()[has].tolist())
            sev_pred.extend(out["severity"].detach().cpu().numpy()[has].tolist())
        attn = out["attention"].detach().cpu().numpy()
        for sid, weights in zip(batch["session_id"], attn):
            for i, w in enumerate(weights):
                attn_rows.append({"session_id": sid, "segment_index": i, "attention_weight": float(w)})
    y_prob = np.vstack(y_prob); domain_prob = np.vstack(domain_prob)
    cm = confusion_matrix(y_true, np.argmax(y_prob, axis=1)).tolist()
    metrics = classification_metrics(np.array(y_true), y_prob)
    metrics.update({f"severity_{k}": v for k, v in regression_metrics(sev_true, sev_pred).items()})
    metrics.update({f"domain_{k}": v for k, v in classification_metrics(np.array(domains), domain_prob).items() if k in ["accuracy", "uar"]})
    metrics["confusion_matrix"] = cm
    if save_attention_path:
        Path(save_attention_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(attn_rows).to_csv(save_attention_path, index=False)
    return metrics


def save_confusion_matrix(cm, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(np.asarray(cm))
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close(fig)
    except Exception:
        # Fallback: save the matrix as text if plotting backend is unavailable.
        np.savetxt(str(out_path).replace(".png", ".txt"), np.asarray(cm), fmt="%d")
