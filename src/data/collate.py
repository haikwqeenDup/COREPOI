from __future__ import annotations

from typing import Any, Dict, List

import torch


def _stack_or_tensorize(values: List[Any]) -> torch.Tensor:
    """Robust collate helper.

    Our datasets return a dict of items. Most are already torch.Tensors, but some
    fields (e.g. user id) might be plain Python ints.

    PyTorch's default collate can handle ints, but since we use a custom collate
    we must explicitly support non-tensor values.
    """
    v0 = values[0]
    if isinstance(v0, torch.Tensor):
        return torch.stack(values, dim=0)
    # Handle python ints / floats / lists / numpy scalars / numpy arrays
    return torch.as_tensor(values)


def collate_pretrain(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate for MLM pretrain.

    Returns a dict of batched tensors.
    Shapes (typical):
      input_tokens:   [B, L]
      tod/dow:        [B, L]
      attn_mask:      [B, L]
      target_poi/cat: [B, L]
      target_regions: [B, S, L]
      user:           [B]
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    out: Dict[str, torch.Tensor] = {}
    keys = list(batch[0].keys())
    for k in keys:
        out[k] = _stack_or_tensorize([b[k] for b in batch])
    return out


def collate_finetune(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate for next-POI fine-tune."""
    if len(batch) == 0:
        raise ValueError("Empty batch")

    out: Dict[str, torch.Tensor] = {}
    keys = list(batch[0].keys())
    for k in keys:
        out[k] = _stack_or_tensorize([b[k] for b in batch])
    return out
