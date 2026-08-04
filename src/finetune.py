from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_finetune import NextPOIDataset
from src.data.collate import collate_finetune
from src.models.model import STBert, STBertConfig
from src.utils.seed import seed_everything
from src.utils.metrics import topk_and_mrr, format_metrics
from src.utils.checkpoint import save_checkpoint, load_checkpoint


def mask_special_logits(logits: torch.Tensor, num_special: int = 3) -> torch.Tensor:
    # avoid predicting PAD/MASK/CLS
    logits = logits.clone()
    logits[:, :num_special] = -1e9
    return logits


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_metrics = []
    for batch in tqdm(loader, desc="Val", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)
        logits_poi, _, _ = model.forward_next(
            batch["input_tokens"], batch["tod"], batch["dow"], batch["attn_mask"],
            batch["user"], batch["query_tod"], batch["query_dow"],
            causal=True,
        )
        logits_poi = mask_special_logits(logits_poi, num_special=3)
        metrics = topk_and_mrr(logits_poi, batch["target_poi"], topk=(1, 5, 10, 20))
        all_metrics.append(metrics)

    # average
    if len(all_metrics) == 0:
        return {"acc@1": 0.0, "acc@5": 0.0, "acc@10": 0.0, "acc@20": 0.0, "mrr": 0.0}
    out = {}
    for k in all_metrics[0].keys():
        out[k] = sum(m[k] for m in all_metrics) / len(all_metrics)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_path", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--exp_name", type=str, default="planC_finetune")
    ap.add_argument("--init_from", type=str, default=None, help="pretrain ckpt path")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_len", type=int, default=None)

    # model (must match pretrain if loading)
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
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")

    # loss weights
    ap.add_argument("--lambda_cat", type=float, default=0.2)
    ap.add_argument("--lambda_region", type=float, default=0.2)
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    # early stop
    ap.add_argument("--patience", type=int, default=10)

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
    num_tod = int(cache["num_tod_bins"]) + 1
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

    if args.init_from:
        ckpt = load_checkpoint(args.init_from, map_location="cpu")
        model.load_state_dict(ckpt.model_state, strict=True)
        # 【新增】关键一步：继承预训练的输出层权重
        model.next_poi.weight.data.copy_(model.mlm_poi.weight.data)
        model.next_poi.bias.data.copy_(model.mlm_poi.bias.data)

        # 如果开启了 tie_weights，确保 next_poi 也绑定
        if model.cfg.tie_weights:
            model.next_poi.weight = model.poi_emb.weight
        print(f"Loaded pretrain weights from {args.init_from}")

    train_ds = NextPOIDataset(
        cache["splits"]["train"],
        token2cat=cache["token2cat"],
        token2regions=cache["token2regions"],
        max_len=max_len,
        pad_token=PAD,
        cls_token=CLS,
        augment=True,
    )
    val_ds = NextPOIDataset(
        cache["splits"]["val"],
        token2cat=cache["token2cat"],
        token2regions=cache["token2regions"],
        max_len=max_len,
        pad_token=PAD,
        cls_token=CLS,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_finetune, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_finetune, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{args.exp_name}.pt"

    best_mrr = -1.0
    bad = 0

    print(f"[Finetune] device={device}, train_samples={len(train_ds)}, val_samples={len(val_ds)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")
        tot_loss = 0.0
        n = 0

        for batch in pbar:
            for k in batch:
                batch[k] = batch[k].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                logits_poi, logits_cat, logits_regs = model.forward_next(
                    batch["input_tokens"], batch["tod"], batch["dow"], batch["attn_mask"],
                    batch["user"], batch["query_tod"], batch["query_dow"],
                    causal=True,
                )
                logits_poi = mask_special_logits(logits_poi, num_special=3)

                # label smoothing
                if args.label_smoothing and args.label_smoothing > 0:
                    logp = F.log_softmax(logits_poi, dim=-1)
                    nll = -logp.gather(1, batch["target_poi"].view(-1, 1)).squeeze(1)
                    smooth = -logp.mean(dim=-1)
                    loss_poi = ((1.0 - args.label_smoothing) * nll + args.label_smoothing * smooth).mean()
                else:
                    loss_poi = F.cross_entropy(logits_poi, batch["target_poi"])

                loss_cat = F.cross_entropy(logits_cat, batch["target_cat"])
                loss_reg = 0.0
                for s, reg_logits in enumerate(logits_regs):
                    loss_reg = loss_reg + F.cross_entropy(reg_logits, batch["target_regions"][:, s])

                loss = loss_poi + args.lambda_cat * loss_cat + args.lambda_region * loss_reg

            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item()
            n += 1
            pbar.set_postfix({"loss": f"{tot_loss/max(n,1):.4f}", "poi": f"{loss_poi.item():.3f}", "cat": f"{loss_cat.item():.3f}"})

        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: VAL {format_metrics(val_metrics)}")

        if val_metrics["mrr"] > best_mrr + 1e-6:
            best_mrr = val_metrics["mrr"]
            bad = 0
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_mrr,
                hparams={
                    "mode": "finetune",
                    "cfg": cfg.__dict__,
                    "lambda_cat": args.lambda_cat,
                    "lambda_region": args.lambda_region,
                    "label_smoothing": args.label_smoothing,
                    "init_from": args.init_from,
                },
            )
            print(f"Saved best finetune ckpt to {ckpt_path} (best_mrr={best_mrr:.4f})")
        else:
            bad += 1
            print(f"No improvement. bad={bad}/{args.patience}")
            if bad >= args.patience:
                print("Early stop.")
                break

    print("Done.")


if __name__ == "__main__":
    main()
