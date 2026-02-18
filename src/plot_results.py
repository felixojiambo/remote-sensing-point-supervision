import csv
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS = Path("reports/results.csv")

def main():
    rows = []
    with open(RESULTS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Experiment A plot: mIoU vs K (uniform)
    a = [r for r in rows if r["experiment"] == "A_K_sweep"]
    a = sorted(a, key=lambda r: int(r["K"]))
    ks = [int(r["K"]) for r in a]
    miou = [float(r["best_miou"]) for r in a]

    plt.figure()
    plt.plot(ks, miou, marker="o")
    plt.title("Experiment A: mIoU vs Point Density (K)")
    plt.xlabel("K (points per image)")
    plt.ylabel("Best mIoU")
    plt.grid(True)
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/miou_vs_k.png", dpi=200)
    plt.close()

    # Experiment B: uniform vs class-balanced at fixed K
    b = [r for r in rows if r["experiment"] == "B_strategy"]
    labels = [r["strategy"] for r in b]
    vals = [float(r["best_miou"]) for r in b]

    plt.figure()
    plt.bar(labels, vals)
    plt.title("Experiment B: Sampling Strategy Comparison (mIoU)")
    plt.xlabel("Strategy")
    plt.ylabel("Best mIoU")
    plt.savefig("reports/strategy_miou.png", dpi=200)
    plt.close()

    print("Saved plots to reports/")

if __name__ == "__main__":
    main()
