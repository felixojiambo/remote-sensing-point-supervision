import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import LoveDADataset
from src.utils import build_transforms

def main():
    root = "data/raw/LoveDA"  # adjust if needed
    img_size = 512

    ds = LoveDADataset(
        root=root,
        split="Train",
        domain="Urban",
        img_size=img_size,
        transforms=build_transforms(img_size=img_size, train=True),
    )

    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    images, masks = next(iter(dl))
    print("images:", images.shape, images.dtype)  # (B, 3, H, W)
    print("masks:", masks.shape, masks.dtype)     # (B, H, W)

    # Sanity plot first sample
    img = images[0].permute(1, 2, 0).cpu().numpy()
    mask = masks[0].cpu().numpy()

    plt.figure()
    plt.title("Image")
    plt.imshow(img)
    plt.axis("off")

    plt.figure()
    plt.title("Mask (class indices)")
    plt.imshow(mask)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
