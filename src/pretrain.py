from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_pretrain import MaskedPretrainDataset
from src.data.collate import collate_pretrain
from src.models.model import STBert, STBertConfig
from src.utils.seed import seed_everything
from src.utils.checkpoint import save_checkpoint


def compute_losses(
    poi_logits, cat_logits, reg_logits_list,
    target_poi, target_cat, target_regions,
    lambda_cat: float, lambda_region: float,
    ignore_index: int = -100,
):
    # poi
    loss_poi = F.cross_entropy(poi_logits.view(-1, poi_logits.size(-1)), target_poi.view(-1), ignore_index=ignore_index)
    loss_cat = F.cross_entropy(cat_logits.view(-1, cat_logits.size(-1)), target_cat.view(-1), ignore_index=ignore_index)
    loss_reg = 0.0
    # target_regions: [B,S,L]
    for s, reg_logits in enumerate(reg_logits_list):
        tgt = target_regions[:, s, :]  # [B,L]
        loss_reg = loss_reg + F.cross_entropy(reg_logits.view(-1, reg_logits.size(-1)), tgt.reshape(-1), ignore_index=ignore_index)

    loss = loss_poi + float(lambda_cat) * loss_cat + float(lambda_region) * loss_reg
    return loss, loss_poi.detach(), loss_cat.detach(), torch.tensor(loss_reg).detach()


@torch.no_grad()
def evaluate(model, loader, device, lambda_cat: float, lambda_region: float):
    model.eval()
    tot = 0.0
    tot_poi = 0.0
    tot_cat = 0.0
    tot_reg = 0.0
    n = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)
        poi_logits, cat_logits, reg_logits = model.forward_mlm(
            batch["input_tokens"], batch["tod"], batch["dow"], batch["attn_mask"], batch["user"]
        )
        loss, lp, lc, lr = compute_losses(
            poi_logits, cat_logits, reg_logits,
            batch["target_poi"], batch["target_cat"], batch["target_regions"],
            lambda_cat=lambda_cat, lambda_region=lambda_region
        )
        tot += loss.item()
        tot_poi += lp.item()
        tot_cat += lc.item()
        tot_reg += lr.item()
        n += 1
    if n == 0:
        return {"loss": 0.0, "poi": 0.0, "cat": 0.0, "reg": 0.0}
    return {"loss": tot / n, "poi": tot_poi / n, "cat": tot_cat / n, "reg": tot_reg / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_path", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--exp_name", type=str, default="planC_pretrain")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_len", type=int, default=None, help="override cache max_len")
    ap.add_argument("--mask_prob", type=float, default=0.25)
    ap.add_argument("--window_stride", type=int, default=64)

    # model hparams
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--user_dim", type=int, default=128)
    ap.add_argument("--pos2d_freq", type=int, default=8)
    ap.add_argument("--tie_weights", action="store_true")

    # training
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")

    # loss weights
    ap.add_argument("--lambda_cat", type=float, default=0.2)
    ap.add_argument("--lambda_region", type=float, default=0.2)

    ap.add_argument("--patience", type=int, default=8)

    args = ap.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache_path, map_location="cpu")

    PAD = cache["special_tokens"]["pad"]
    MASK = cache["special_tokens"]["mask"]
    CLS = cache["special_tokens"]["cls"]

    max_len = int(args.max_len) if args.max_len is not None else int(cache["max_len"])
    vocab_size = int(cache["vocab_size"])
    num_users = int(cache["num_users"])
    num_cats = int(cache["num_cats"])
    num_tod = int(cache["num_tod_bins"]) + 1  # shift(+1) + 0 unknown
    num_dow = 8
    region_vocab_sizes = [int(v) for v in cache["region_vocab_sizes"]]

    cfg = STBertConfig(
        vocab_size=vocab_size,
        num_users=num_users,
        num_cats=num_cats,
        num_tod=num_tod,
        num_dow=num_dow,
        region_vocab_sizes=region_vocab_sizes,
        max_len=max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        user_dim=args.user_dim,
        pos2d_freq=args.pos2d_freq,
        tie_weights=args.tie_weights,
    )

    model = STBert(
        cfg,
        token2cat=cache["token2cat"],
        token2regions=cache["token2regions"],
        token2xy=cache["token2xy"],
        pad_token=PAD,
        mask_token=MASK,
        cls_token=CLS,
    ).to(device)

    train_ds = MaskedPretrainDataset(
        cache["splits"]["train"],
        token2cat=cache["token2cat"],
        token2regions=cache["token2regions"],
        max_len=max_len,
        mask_prob=args.mask_prob,
        pad_token=PAD,
        mask_token=MASK,
        cls_token=CLS,
        vocab_size=vocab_size,
        window_stride=args.window_stride,
        seed=args.seed,
    )
    val_ds = MaskedPretrainDataset(
        cache["splits"]["val"],
        token2cat=cache["token2cat"],
        token2regions=cache["token2regions"],
        max_len=max_len,
        mask_prob=args.mask_prob,
        pad_token=PAD,
        mask_token=MASK,
        cls_token=CLS,
        vocab_size=vocab_size,
        window_stride=args.window_stride,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_pretrain, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_pretrain, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{args.exp_name}.pt"

    best_val = float("inf")
    bad = 0

    print(f"[Pretrain] device={device}, vocab={vocab_size}, users={num_users}, cats={num_cats}, regions={region_vocab_sizes}, max_len={max_len}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")
        tot = 0.0
        n = 0
        for batch in pbar:
            for k in batch:
                batch[k] = batch[k].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                poi_logits, cat_logits, reg_logits = model.forward_mlm(
                    batch["input_tokens"], batch["tod"], batch["dow"], batch["attn_mask"], batch["user"]
                )
                loss, lp, lc, lr = compute_losses(
                    poi_logits, cat_logits, reg_logits,
                    batch["target_poi"], batch["target_cat"], batch["target_regions"],
                    lambda_cat=args.lambda_cat, lambda_region=args.lambda_region
                )

            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tot += loss.item()
            n += 1
            pbar.set_postfix({"loss": f"{tot/max(n,1):.4f}", "poi": f"{lp.item():.3f}", "cat": f"{lc.item():.3f}", "reg": f"{lr.item():.3f}"})

        val = evaluate(model, val_loader, device, args.lambda_cat, args.lambda_region)
        print(f"Epoch {epoch}: val_loss={val['loss']:.4f} (poi={val['poi']:.4f}, cat={val['cat']:.4f}, reg={val['reg']:.4f})")

        if val["loss"] + 1e-6 < best_val:
            best_val = val["loss"]
            bad = 0
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=-best_val,  # higher is better
                hparams={
                    "mode": "pretrain",
                    "cfg": cfg.__dict__,
                    "lambda_cat": args.lambda_cat,
                    "lambda_region": args.lambda_region,
                    "mask_prob": args.mask_prob,
                },
            )
            print(f"Saved best pretrain ckpt to {ckpt_path} (best_val_loss={best_val:.4f})")
        else:
            bad += 1
            print(f"No improvement. bad={bad}/{args.patience}")
            if bad >= args.patience:
                print("Early stop.")
                break

    print("Done.")


if __name__ == "__main__":
    main()
