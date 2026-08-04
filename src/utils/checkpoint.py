from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Checkpoint:
    model_state: dict
    optimizer_state: dict | None
    epoch: int
    best_metric: float
    hparams: dict


def save_checkpoint(path: str | Path, *, model, optimizer=None, epoch: int = 0, best_metric: float = 0.0, hparams: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "hparams": hparams or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> Checkpoint:
    payload = torch.load(path, map_location=map_location)
    return Checkpoint(
        model_state=payload["model_state"],
        optimizer_state=payload.get("optimizer_state"),
        epoch=int(payload.get("epoch", 0)),
        best_metric=float(payload.get("best_metric", 0.0)),
        hparams=payload.get("hparams", {}) or {},
    )
