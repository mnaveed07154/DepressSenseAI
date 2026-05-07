import argparse
from src.config import Config
from src.train import main as train_main
from src.dataset import create_demo_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DepressSenseAI runner")
    parser.add_argument("--mode", choices=["demo", "train"], default="demo")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    if args.mode == "demo":
        cfg = Config(demo_mode=True, use_ssl=False, epochs=3, batch_size=4)
        create_demo_metadata(cfg.metadata_csv)
        cfg.save(args.config)
        train_main(args.config)
    else:
        train_main(args.config)
