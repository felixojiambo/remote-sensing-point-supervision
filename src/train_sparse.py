import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.config import NUM_CLASSES, IGNORE_INDEX, SEED, DEFAULT_POINTS
from src.dataset import LoveDADataset
from src.model import build_model
from src.point_sampler import sample_points_batch
from src.losses import partial_ce_loss
from src.metrics import confusion_matrix, miou_from_cm, pixel_accuracy_from_cm
from src.utils import set_seed, build_transforms

import matplotlib.pyplot as plt


def save_sample_viz(run_dir: str, images, mask_full, mask_sparse, pred, epoch: int):
    """
    Save 1 sample visualization: image | sparse points | prediction | full mask
    """
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    # pick first item
    img = images[0].detach().cpu()
    mf = mask_full[0].detach().cpu()
    ms = mask_sparse[0].detach().cpu()
    pr = pred[0].detach().cpu()

    # image is normalized; rescale for viewing
    img = img.permute(1, 2, 0).numpy()
    img_disp = (img - img.min()) / (img.max() - img.min() + 1e-6)

    # sparse points overlay: points where ms != IGNORE_INDEX
    pts = (ms != IGNORE_INDEX).numpy()

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("Image")
    plt.imshow(img_disp)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Sparse points (red)")
    plt.imshow(mf.numpy())
    overlay = plt.imshow(pts, alpha=0.6)
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Prediction")
    plt.imshow(pr.numpy())
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Full mask (GT)")
    plt.imshow(mf.numpy())
    plt.axis("off")

    out_path = os.path.join(run_dir, "samples", f"epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    # --- Config (keep simple for now; can move to YAML later) ---
    run_name = f"sparse_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "run_name": run_name,
        "dataset_root": "data/raw/LoveDA",
        "train_split": "Train",
        "val_split": "Val",
        "domain": "Urban",
        "img_size": 512,
        "batch_size": 2,
        "epochs": 5,
        "lr": 1e-4,
        "points_per_image": DEFAULT_POINTS,
        "sampling_strategy": "uniform",  # change to "class_balanced" for exp
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "seed": SEED,
        "model": {"arch": "unet", "encoder": "resnet34", "pretrained": "imagenet"},
    }

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Run dir:", run_dir)

    # --- Data ---
    train_ds = LoveDADataset(
        root=config["dataset_root"],
        split=config["train_split"],
        domain=config["domain"],
        transforms=build_transforms(img_size=config["img_size"], train=True),
    )
    val_ds = LoveDADataset(
        root=config["dataset_root"],
        split=config["val_split"],
        domain=config["domain"],
        transforms=build_transforms(img_size=config["img_size"], train=False),
    )

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    # --- Model ---
    model = build_model(num_classes=NUM_CLASSES).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_miou = -1.0

    for epoch in range(1, config["epochs"] + 1):
        # ------------------ Train ------------------
        model.train()
        total_loss = 0.0
        steps = 0

        for images, mask_full in train_dl:
            images = images.to(device)
            mask_full = mask_full.to(device)

            mask_sparse = sample_points_batch(
                mask_full,
                K=config["points_per_image"],
                strategy=config["sampling_strategy"],
                num_classes=NUM_CLASSES,
            ).to(device)

            logits = model(images)
            loss = partial_ce_loss(logits, mask_sparse, ignore_index=IGNORE_INDEX)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)

        # ------------------ Validate ------------------
        model.eval()
        cm_total = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long, device=device)

        with torch.no_grad():
            for images, mask_full in val_dl:
                images = images.to(device)
                mask_full = mask_full.to(device)

                logits = model(images)
                pred = torch.argmax(logits, dim=1)  # (B,H,W)

                cm = confusion_matrix(pred, mask_full, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
                cm_total += cm

        miou = miou_from_cm(cm_total)
        acc = pixel_accuracy_from_cm(cm_total)

        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | mIoU={miou:.4f} | acc={acc:.4f}")

        # Save checkpoint each epoch
        ckpt_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "miou": miou,
                "loss": avg_loss,
                "config": config,
            },
            ckpt_path,
        )

        # Save best
        if miou > best_miou:
            best_miou = miou
            best_path = os.path.join(run_dir, "best.pt")
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "miou": miou, "config": config}, best_path)

        # Save 1 sample viz from a single val batch for the report
        with torch.no_grad():
            images, mask_full = next(iter(val_dl))
            images = images.to(device)
            mask_full = mask_full.to(device)

            mask_sparse = sample_points_batch(
                mask_full,
                K=config["points_per_image"],
                strategy=config["sampling_strategy"],
                num_classes=NUM_CLASSES,
            ).to(device)

            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            save_sample_viz(run_dir, images, mask_full, mask_sparse, pred, epoch)

    print("Done. Best mIoU:", best_miou)


if __name__ == "__main__":
    main()
