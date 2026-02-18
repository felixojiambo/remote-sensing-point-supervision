import torch
import torch.nn.functional as F

def partial_ce_loss(
    logits: torch.Tensor,
    mask_sparse: torch.Tensor,
    ignore_index: int = -1,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Partial Cross Entropy over labeled pixels only.

    logits: (B, C, H, W) float
    mask_sparse: (B, H, W) long with ignore_index for unlabeled pixels
    ignore_index: int

    Returns: scalar loss
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be (B,C,H,W). Got {logits.shape}")
    if mask_sparse.dim() != 3:
        raise ValueError(f"mask_sparse must be (B,H,W). Got {mask_sparse.shape}")
    if logits.shape[0] != mask_sparse.shape[0] or logits.shape[2:] != mask_sparse.shape[1:]:
        raise ValueError(f"Shape mismatch: logits {logits.shape}, mask_sparse {mask_sparse.shape}")

    # Per-pixel CE loss map (B,H,W). ignore_index pixels contribute 0 here (but we still mask explicitly)
    loss_map = F.cross_entropy(
        logits,
        mask_sparse.long(),
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )

    labeled = (mask_sparse != ignore_index)
    labeled_count = labeled.sum()

    # Edge case: no labeled pixels => return 0 (safe, does not crash training)
    if labeled_count.item() == 0:
        return loss_map.sum() * 0.0

    return loss_map[labeled].mean()
