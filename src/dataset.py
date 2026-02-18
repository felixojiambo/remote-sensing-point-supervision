import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class LoveDADataset(Dataset):
    """
    Returns:
      image: FloatTensor (3, H, W) normalized
      mask_full: LongTensor (H, W) with values 0..C-1
    """

    def __init__(
        self,
        root: str,
        split: str = "Train",        # Train or Val
        domain: str = "Urban",       # Urban or Rural
        img_size: int = 512,
        transforms=None,
    ):
        self.root = Path(root)
        self.split = split
        self.domain = domain
        self.img_size = img_size
        self.transforms = transforms

        img_dir = self.root / split / domain / "images_png"
        mask_dir = self.root / split / domain / "masks_png"

        if not img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask dir not found: {mask_dir}")

        self.image_paths = sorted([p for p in img_dir.glob("*.png")])
        self.mask_paths = sorted([mask_dir / p.name for p in self.image_paths])

        # Validate masks exist
        missing = [str(m) for m in self.mask_paths if not m.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} mask files, e.g. {missing[0]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # LoveDA images are RGB .png
        image = np.array(Image.open(img_path).convert("RGB"))
        # LoveDA masks are single-channel class indices
        mask = np.array(Image.open(mask_path))

        # Albumentations expects dict with image/mask
        if self.transforms is not None:
            out = self.transforms(image=image, mask=mask)
            image = out["image"]          # torch tensor CHW
            mask = out["mask"]            # torch tensor HW or numpy -> depends on ToTensorV2
        else:
            # fallback (no aug): convert manually
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # Ensure correct dtypes
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()

        image = image.float()
        mask = mask.long()

        return image, mask
