# train.py
"""
Top-level entrypoint for the project.

Usage:
  python train.py --config configs/dev.json

This script loads a JSON config and forwards the values to:
  python -m src.train_sparse --domain ... --img_size ... --epochs ... etc.

Why this exists:
- Keeps submission clean (single train.py entrypoint).
- Lets reviewers run reproducible configs easily.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path


ALLOWED_KEYS = {
    "domain",
    "img_size",
    "batch_size",
    "epochs",
    "lr",
    "K",
    "strategy",
    "seed",
    "num_workers",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train sparse point-supervision model using a JSON config.")
    p.add_argument("--config", type=str, required=True, help="Path to JSON config file (e.g., configs/dev.json)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Validate keys (fail fast on typos)
    unknown = set(cfg.keys()) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}. Allowed keys: {sorted(ALLOWED_KEYS)}")

    # Build command using the current interpreter (venv-safe)
    cmd = [sys.executable, "-m", "src.train_sparse"]

    # Only pass keys that exist in the config
    if "domain" in cfg:
        cmd += ["--domain", str(cfg["domain"])]
    if "img_size" in cfg:
        cmd += ["--img_size", str(cfg["img_size"])]
    if "batch_size" in cfg:
        cmd += ["--batch_size", str(cfg["batch_size"])]
    if "epochs" in cfg:
        cmd += ["--epochs", str(cfg["epochs"])]
    if "lr" in cfg:
        cmd += ["--lr", str(cfg["lr"])]
    if "K" in cfg:
        cmd += ["--K", str(cfg["K"])]
    if "strategy" in cfg:
        cmd += ["--strategy", str(cfg["strategy"])]
    if "seed" in cfg:
        cmd += ["--seed", str(cfg["seed"])]
    if "num_workers" in cfg:
        cmd += ["--num_workers", str(cfg["num_workers"])]

    print("Running:", " ".join(cmd))

    # Ensure project root on PYTHONPATH (so `-m src.train_sparse` resolves consistently)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
