import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms(img_size: int = 512, train: bool = True):
    if train:
        return A.Compose([
            # Albumentations v2: use size=(H, W)
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ColorJitter(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(),
            ToTensorV2(),
        ])
