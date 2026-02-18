import torch
from src.config import IGNORE_INDEX

def sample_points(mask_full: torch.Tensor, K: int, strategy: str = "uniform", num_classes: int = 7) -> torch.Tensor:
    """
    mask_full: (H, W) long with values 0..C-1, and IGNORE_INDEX for ignore pixels
    returns:
      mask_sparse: (H, W) long filled with IGNORE_INDEX except K labeled pixels
    """
    if mask_full.dim() != 2:
        raise ValueError(f"mask_full must be (H,W). Got {mask_full.shape}")

    H, W = mask_full.shape
    total_pixels = H * W

    if K <= 0:
        raise ValueError("K must be > 0")
    if K > total_pixels:
        raise ValueError(f"K={K} exceeds total pixels={total_pixels}")

    # Only sample from valid (non-ignore) pixels
    valid_coords = (mask_full != IGNORE_INDEX).nonzero(as_tuple=False)
    if valid_coords.numel() == 0:
        # No valid labels in this mask; return all ignore
        return torch.full((H, W), IGNORE_INDEX, dtype=torch.long)

    if strategy not in {"uniform", "class_balanced"}:
        raise ValueError("strategy must be one of: uniform, class_balanced")

    mask_sparse = torch.full((H, W), IGNORE_INDEX, dtype=torch.long)

    if strategy == "uniform":
        # sample uniformly among valid pixels
        n = valid_coords.shape[0]
        k_eff = min(K, n)
        idx = torch.randperm(n)[:k_eff]
        chosen = valid_coords[idx]
        mask_sparse[chosen[:, 0], chosen[:, 1]] = mask_full[chosen[:, 0], chosen[:, 1]]
        return mask_sparse

    # class_balanced
    chosen_coords = []
    # gather coords per class
    for c in range(num_classes):
        c_coords = (mask_full == c).nonzero(as_tuple=False)
        if c_coords.numel() > 0:
            chosen_coords.append((c, c_coords))

    if len(chosen_coords) == 0:
        return mask_sparse

    # Allocate points roughly evenly across present classes
    classes_present = len(chosen_coords)
    base = K // classes_present
    remainder = K % classes_present

    picked = []
    for i, (c, coords) in enumerate(chosen_coords):
        k_c = base + (1 if i < remainder else 0)
        if k_c <= 0:
            continue
        n = coords.shape[0]
        k_eff = min(k_c, n)  # safety: if class has too few pixels
        idx = torch.randperm(n)[:k_eff]
        picked.append(coords[idx])

    if len(picked) == 0:
        return mask_sparse

    picked = torch.cat(picked, dim=0)

    # If due to tiny classes we picked fewer than K, top up uniformly from valid pixels
    if picked.shape[0] < min(K, valid_coords.shape[0]):
        need = min(K, valid_coords.shape[0]) - picked.shape[0]
        # remove already picked from candidates (simple approach: use a set on CPU)
        picked_cpu = set((int(r), int(c)) for r, c in picked.cpu().tolist())
        candidates = [(int(r), int(c)) for r, c in valid_coords.cpu().tolist() if (int(r), int(c)) not in picked_cpu]
        if len(candidates) > 0:
            perm = torch.randperm(len(candidates))[:min(need, len(candidates))]
            extra = torch.tensor([candidates[i] for i in perm.tolist()], dtype=torch.long)
            picked = torch.cat([picked, extra], dim=0)

    mask_sparse[picked[:, 0], picked[:, 1]] = mask_full[picked[:, 0], picked[:, 1]]
    return mask_sparse


def sample_points_batch(mask_full_bhw: torch.Tensor, K: int, strategy: str, num_classes: int = 7) -> torch.Tensor:
    """
    mask_full_bhw: (B,H,W)
    returns: (B,H,W) sparse
    """
    if mask_full_bhw.dim() != 3:
        raise ValueError(f"Expected (B,H,W). Got {mask_full_bhw.shape}")

    out = []
    for b in range(mask_full_bhw.shape[0]):
        out.append(sample_points(mask_full_bhw[b], K=K, strategy=strategy, num_classes=num_classes))
    return torch.stack(out, dim=0)
