import torch

def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = -1) -> torch.Tensor:
    """
    pred: (B,H,W) long
    target: (B,H,W) long
    Returns: (C,C) confusion matrix
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch pred {pred.shape} vs target {target.shape}")

    pred = pred.view(-1)
    target = target.view(-1)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long, device=pred.device)

    idx = target * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm

def miou_from_cm(cm: torch.Tensor) -> float:
    """
    cm: (C,C)
    """
    cm = cm.float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.nan)
    return float(torch.nanmean(iou).item())

def pixel_accuracy_from_cm(cm: torch.Tensor) -> float:
    cm = cm.float()
    correct = torch.diag(cm).sum()
    total = cm.sum()
    if total.item() == 0:
        return 0.0
    return float((correct / total).item())
