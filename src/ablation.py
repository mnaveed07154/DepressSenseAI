from pathlib import Path
import copy
import pandas as pd
from .config import Config
from .train import main as train_main


def run_ablation(base_config_path=None):
    base = Config.load(base_config_path) if base_config_path else Config()
    experiments = {
        "full": {},
        "no_ssl_pretrained": {"use_ssl": False},
        "low_domain_weight": {"beta_domain": 0.0},
        "no_severity_weight": {"alpha_severity": 0.0},
        "small_fusion_dim": {"fusion_dim": 128},
    }
    rows = []
    for name, updates in experiments.items():
        cfg = copy.deepcopy(base)
        for k, v in updates.items(): setattr(cfg, k, v)
        cfg.results_dir = str(Path(base.results_dir) / "ablation" / name)
        cfg_path = Path(cfg.results_dir) / "config.json"
        cfg.save(cfg_path)
        metrics = train_main(str(cfg_path))
        rows.append({"experiment": name, **{k:v for k,v in metrics.items() if k != "confusion_matrix"}})
    out = Path(base.results_dir)/"metrics"/"ablation_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return rows

if __name__ == "__main__":
    run_ablation()
