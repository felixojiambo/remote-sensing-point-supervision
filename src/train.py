import os
import yaml
import torch
from src.config import DEVICE

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)

    print("=================================")
    print("Starting Experiment:", config["experiment_name"])
    print("Using device:", DEVICE)
    print("Dataset:", config["dataset"]["name"])
    print("Points per image:", config["dataset"]["points_per_image"])
    print("=================================")

    # Create run directory
    run_dir = os.path.join("runs", config["experiment_name"])
    os.makedirs(run_dir, exist_ok=True)

    print("Run directory created:", run_dir)
    print("Phase 1 setup successful.")
    print("Ready for Phase 2 (dataset implementation).")

if __name__ == "__main__":
    main("configs/base.yaml")
