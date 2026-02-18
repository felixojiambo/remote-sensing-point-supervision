import torch
from torch.utils.data import DataLoader

from src.dataset import LoveDADataset
from src.utils import build_transforms
from src.model import build_model
from src.config import NUM_CLASSES

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = LoveDADataset(
        root="data/raw/LoveDA",
        split="Train",
        domain="Urban",
        transforms=build_transforms(img_size=512, train=False),
    )
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    images, masks = next(iter(dl))
    images = images.to(device)
    masks = masks.to(device)

    model = build_model(num_classes=NUM_CLASSES).to(device)
    logits = model(images)

    print("images:", images.shape)
    print("masks: ", masks.shape)
    print("logits:", logits.shape)

    assert logits.shape[0] == images.shape[0]
    assert logits.shape[1] == NUM_CLASSES
    assert logits.shape[2:] == masks.shape[1:], "Logits spatial dims must match mask"

    print("OK: model forward output dims are correct.")

if __name__ == "__main__":
    main()
