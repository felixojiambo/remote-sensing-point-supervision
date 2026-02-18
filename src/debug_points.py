import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import LoveDADataset
from src.utils import build_transforms
from src.point_sampler import sample_points
from src.config import IGNORE_INDEX, NUM_CLASSES

def points_overlay(mask_full: np.ndarray, mask_sparse: np.ndarray):
    # mask_sparse: IGNORE_INDEX everywhere except points
    overlay = np.zeros((*mask_full.shape, 4), dtype=np.float32)  # RGBA
    pts = (mask_sparse != IGNORE_INDEX)
    overlay[pts] = np.array([1.0, 0.0, 0.0, 0.9])  # red points
    return overlay

def main():
    root = "data/raw/LoveDA"  # adjust if needed
    ds = LoveDADataset(
        root=root,
        split="Train",
        domain="Urban",
        transforms=build_transforms(img_size=512, train=False),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    image, mask_full = next(iter(dl))
    image = image[0]
    mask_full = mask_full[0]

    K = 100
    sparse_u = sample_points(mask_full, K=K, strategy="uniform", num_classes=NUM_CLASSES)
    sparse_cb = sample_points(mask_full, K=K, strategy="class_balanced", num_classes=NUM_CLASSES)

    img = image.permute(1, 2, 0).cpu().numpy()
    # image is normalized; for display clamp to 0..1 after shifting into range
    img_disp = np.clip((img - img.min()) / (img.max() - img.min() + 1e-6), 0, 1)

    mf = mask_full.cpu().numpy()
    su = sparse_u.cpu().numpy()
    scb = sparse_cb.cpu().numpy()

    plt.figure()
    plt.title("Image")
    plt.imshow(img_disp)
    plt.axis("off")

    plt.figure()
    plt.title("Full mask (0..6, IGNORE=-1)")
    plt.imshow(mf)
    plt.axis("off")

    plt.figure()
    plt.title(f"Uniform points overlay (K={K})")
    plt.imshow(mf)
    plt.imshow(points_overlay(mf, su))
    plt.axis("off")

    plt.figure()
    plt.title(f"Class-balanced points overlay (K={K})")
    plt.imshow(mf)
    plt.imshow(points_overlay(mf, scb))
    plt.axis("off")

    # Unit-test-like checks
    assert (sparse_u != IGNORE_INDEX).sum().item() <= K
    assert (sparse_cb != IGNORE_INDEX).sum().item() <= K
    assert torch.all((sparse_u == IGNORE_INDEX) | (sparse_u == mask_full))
    assert torch.all((sparse_cb == IGNORE_INDEX) | (sparse_cb == mask_full))

    print("OK: sparse masks generated and validated.")
    plt.show()

if __name__ == "__main__":
    main()
