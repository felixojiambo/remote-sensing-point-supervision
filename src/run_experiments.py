import sys
import csv
import os
import subprocess
from pathlib import Path

RESULTS_PATH = Path("reports/results.csv")

def ensure_results_header():
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.exists():
        with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "experiment", "domain", "img_size", "epochs", "batch_size",
                "K", "strategy", "best_miou", "best_acc", "run_dir"
            ])

def run_one(experiment, domain, img_size, epochs, batch_size, K, strategy):
    cmd = [
        sys.executable, "-m", "src.train_sparse",
        "--domain", domain,
        "--img_size", str(img_size),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--K", str(K),
        "--strategy", strategy,
    ]
    print("Running:", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True)

    # train_sparse prints: Done. Best mIoU: <float>
    best_line = [l for l in out.splitlines() if "Done. Best mIoU:" in l][-1]
    best_miou = float(best_line.split(":")[-1].strip())

    # run_dir printed: Run dir: runs\...
    run_line = [l for l in out.splitlines() if l.startswith("Run dir:")][-1]
    run_dir = run_line.split("Run dir:")[-1].strip()

    # We also printed acc per epoch; take last epoch acc as proxy
    last_epoch = [l for l in out.splitlines() if l.startswith("Epoch")][-1]
    # format: Epoch 002 | loss=... | mIoU=... | acc=...
    best_acc = float(last_epoch.split("acc=")[-1])

    return best_miou, best_acc, run_dir

def main():
    ensure_results_header()

    domain = "Urban"
    img_size = 128
    epochs = 2
    batch_size = 8

    # Experiment A: K sweep
    for K in [10, 50, 200, 1000]:
        best_miou, best_acc, run_dir = run_one("A_K_sweep", domain, img_size, epochs, batch_size, K, "uniform")
        with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["A_K_sweep", domain, img_size, epochs, batch_size, K, "uniform", best_miou, best_acc, run_dir])

    # Experiment B: sampling strategy at fixed K
    K_fixed = 200
    for strategy in ["uniform", "class_balanced"]:
        best_miou, best_acc, run_dir = run_one("B_strategy", domain, img_size, epochs, batch_size, K_fixed, strategy)
        with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["B_strategy", domain, img_size, epochs, batch_size, K_fixed, strategy, best_miou, best_acc, run_dir])

    print("Wrote:", RESULTS_PATH)

if __name__ == "__main__":
    main()
