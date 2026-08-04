from __future__ import annotations
import torch


@torch.no_grad()
def topk_and_mrr(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5, 10, 20)) -> dict:
    """
    logits: [B, V]
    targets: [B] long
    Returns dict with acc@k and mrr
    """
    assert logits.ndim == 2
    assert targets.ndim == 1
    B, V = logits.shape
    device = logits.device

    # ranks (1-based)
    # Use argsort descending; topk only for acc.
    metrics = {}
    for k in topk:
        k = min(int(k), V)
        pred = logits.topk(k, dim=1).indices  # [B,k]
        hit = (pred == targets.unsqueeze(1)).any(dim=1).float()
        metrics[f"acc@{k}"] = hit.mean().item()

    # MRR
    sorted_idx = logits.argsort(dim=1, descending=True)  # [B,V]
    # find rank of target
    # Create positions [B,V] of sorted indices; but that's O(BV). Better: use scatter.
    inv_rank = torch.empty((B, V), device=device, dtype=torch.long)
    inv_rank.scatter_(1, sorted_idx, torch.arange(V, device=device).unsqueeze(0).expand(B, -1))
    rank0 = inv_rank.gather(1, targets.view(-1, 1)).squeeze(1)  # 0-based rank
    mrr = (1.0 / (rank0.float() + 1.0)).mean().item()
    metrics["mrr"] = mrr
    return metrics


def format_metrics(m: dict) -> str:
    keys = [k for k in m.keys() if k.startswith("acc@")]
    # sort by k
    keys = sorted(keys, key=lambda x: int(x.split("@")[1]))
    parts = [f"{k.upper()}: {m[k]:.4f}" for k in keys]
    parts.append(f"MRR: {m['mrr']:.4f}")
    return " | ".join(parts)
