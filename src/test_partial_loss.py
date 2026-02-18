import torch
from src.losses import partial_ce_loss

def main():
    torch.manual_seed(0)

    B, C, H, W = 1, 3, 4, 4
    ignore = -1

    # Random logits with gradient
    logits = torch.randn(B, C, H, W, requires_grad=True)

    # Sparse mask: all ignore except 2 labeled pixels
    mask = torch.full((B, H, W), ignore, dtype=torch.long)
    mask[0, 1, 1] = 2
    mask[0, 2, 3] = 1

    loss = partial_ce_loss(logits, mask, ignore_index=ignore)
    loss.backward()

    # Gradient sanity checks:
    # For CE, grad exists everywhere in logits (softmax coupling),
    # BUT the loss should only be computed from labeled pixels.
    # We verify this by comparing with loss computed from only those pixels manually.
    with torch.no_grad():
        # manual: select only labeled locations and compute CE on flattened logits
        coords = torch.nonzero(mask != ignore, as_tuple=False)
        # coords: (n, 3) => (b,y,x)
        selected_logits = []
        selected_targets = []
        for b, y, x in coords:
            selected_logits.append(logits.detach()[b, :, y, x].unsqueeze(0))  # (1,C)
            selected_targets.append(mask[b, y, x].unsqueeze(0))               # (1,)
        selected_logits = torch.cat(selected_logits, dim=0)                  # (n,C)
        selected_targets = torch.cat(selected_targets, dim=0)                # (n,)

        manual = torch.nn.functional.cross_entropy(selected_logits, selected_targets, reduction="mean")

    print("partial_ce_loss:", float(loss.detach()))
    print("manual_ce_loss: ", float(manual.detach()))
    assert torch.allclose(loss.detach(), manual.detach(), atol=1e-6), "Loss mismatch vs manual labeled-only CE"

    # Edge case: no labeled pixels
    logits2 = torch.randn(B, C, H, W, requires_grad=True)
    mask2 = torch.full((B, H, W), ignore, dtype=torch.long)
    loss2 = partial_ce_loss(logits2, mask2, ignore_index=ignore)
    loss2.backward()
    print("no-label loss:", float(loss2.detach()))
    assert float(loss2.detach()) == 0.0, "Expected 0 loss when no labeled pixels"

    print("OK: partial CE behaves as labeled-only and is edge-safe.")

if __name__ == "__main__":
    main()
