from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F  # 用于计算 log_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_finetune import NextPOIDataset
from src.data.collate import collate_finetune
from src.models.model import STBert, STBertConfig
from src.utils.metrics import topk_and_mrr, format_metrics
from src.utils.checkpoint import load_checkpoint


def mask_special_logits(logits: torch.Tensor, num_special: int = 3) -> torch.Tensor:
    logits = logits.clone()
    logits[:, :num_special] = -1e9
    return logits


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    # paper-style spatio-temporal rerank (user-conditioned)
    user_period_pois: dict | None = None,
    user_region_pois: dict | None = None,
    topk_region: int = 40,
    topm_time: int = 1,
    gamma: float = 0.0,
    # [新增开关]
    disable_region_rerank: bool = False,
):
    """
    Args:
        disable_region_rerank: 如果为 True，则忽略 Region 预测带来的约束 (Sr set)，
                               仅使用 Time 约束 (St set)。用于 w/o Aux Tasks 场景。
    """
    model.eval()
    metrics_sum = None
    n = 0

    token2cat = model.token2cat
    token2region0 = model._get_token2region(0)

    # helper: build time candidates around query_tod
    def _time_candidates(q: int, m: int, max_t: int) -> list[int]:
        if q <= 0 or m <= 0:
            return []
        out = [q]
        step = 1
        while len(out) < m:
            left = q - step
            right = q + step
            if left > 0:
                out.append(left)
                if len(out) >= m:
                    break
            if right <= max_t:
                out.append(right)
            step += 1
            if left <= 0 and right > max_t:
                break
        return out[:m]

    max_tod = int(model.cfg.num_tod) - 1

    for batch in tqdm(loader, desc="Eval", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)

        poi_logits, cat_logits, reg_logits_list = model.forward_next(
            batch["input_tokens"],
            batch["tod"],
            batch["dow"],
            batch["attn_mask"],
            batch["user"],
            batch["query_tod"],
            batch["query_dow"],
            causal=True,
        )

        # ---- (1) optional: old consistency rerank (global mapping) ----
        if alpha != 0.0:
            region_log_probs = F.log_softmax(reg_logits_list[0], dim=-1)
            poi_region_bias = region_log_probs[:, token2region0]
            poi_logits = poi_logits + alpha * poi_region_bias

        if beta != 0.0:
            cat_log_probs = F.log_softmax(cat_logits, dim=-1)
            poi_cat_bias = cat_log_probs[:, token2cat]
            poi_logits = poi_logits + beta * poi_cat_bias

        # ---- (2) paper-style user-conditioned spatio-temporal rerank ----
        # 逻辑修改：只要 gamma != 0 且 (有Region表 或 有Time表)，就尝试进入逻辑
        # 具体使用哪个表，由 disable_region_rerank 和 数据是否为 None 共同决定
        if gamma != 0.0 and (user_period_pois is not None or user_region_pois is not None):
            
            # --- 准备 Region 候选项 (仅当未禁用且有数据时) ---
            region_topk_cpu = None
            if not disable_region_rerank and user_region_pois is not None:
                # Top-K regions from model prediction
                K = min(int(topk_region), reg_logits_list[0].shape[-1])
                region_topk = torch.topk(reg_logits_list[0], k=K, dim=-1).indices  # [B,K]
                region_topk_cpu = region_topk.detach().cpu().tolist()

            # --- 准备 Time 候选项 (始终可用，因为 query_tod 是输入) ---
            query_tod = batch["query_tod"].detach().cpu().tolist()

            # tiered boosts calculation
            # 如果只用了 Time，那么 Sr 永远为空，Srt 永远为空，只有 gt (gamma time) 生效
            # 这符合 "只使用 Time inference" 的预期
            gr = float(gamma)
            gt = float(gamma)
            gb = float(gr + gt)

            for i, uidx in enumerate(batch["user"].detach().cpu().tolist()):
                uidx = int(uidx)

                # 1. 构建 Region Set (Sr)
                Sr = set()
                if region_topk_cpu is not None: # 如果被禁用，这里是 None，Sr 保持为空
                    u_reg = user_region_pois.get(uidx, {})
                    for r in region_topk_cpu[i]:
                        arr = u_reg.get(int(r))
                        if arr is not None and len(arr) > 0:
                            Sr.update(arr.tolist() if hasattr(arr, "tolist") else list(arr))

                # 2. 构建 Time Set (St)
                St = set()
                if user_period_pois is not None:
                    u_tp = user_period_pois.get(uidx, {})
                    for t in _time_candidates(int(query_tod[i]), int(topm_time), max_tod):
                        arr = u_tp.get(int(t))
                        if arr is not None and len(arr) > 0:
                            St.update(arr.tolist() if hasattr(arr, "tolist") else list(arr))

                # 如果两个集合都空，跳过
                if not Sr and not St:
                    continue

                # Promote POIs in Sr (Region)
                if Sr:
                    idx_r = torch.as_tensor(list(Sr), device=device, dtype=torch.long)
                    poi_logits[i, idx_r] = poi_logits[i, idx_r] + gr

                # Promote POIs in St (Time)
                if St:
                    idx_t = torch.as_tensor(list(St), device=device, dtype=torch.long)
                    poi_logits[i, idx_t] = poi_logits[i, idx_t] + gt

                # Promote POIs in Sr ∩ St (Both)
                # 如果 Sr 为空(被禁用)，则交集也为空，这里自动不执行
                if Sr and St:
                    Srt = Sr.intersection(St)
                    if Srt:
                        extra = gb - gr - gt
                        if extra != 0.0:
                            idx_rt = torch.as_tensor(list(Srt), device=device, dtype=torch.long)
                            poi_logits[i, idx_rt] = poi_logits[i, idx_rt] + extra

        logits = mask_special_logits(poi_logits, num_special=3)

        mtr = topk_and_mrr(logits, batch["target_poi"], topk=(1, 5, 10, 20))
        if metrics_sum is None:
            metrics_sum = {k: 0.0 for k in mtr.keys()}
        for k in mtr:
            metrics_sum[k] += mtr[k]
        n += 1

    if n == 0:
        return {"acc@1": 0.0, "acc@5": 0.0, "acc@10": 0.0, "acc@20": 0.0, "mrr": 0.0}
    return {k: v / n for k, v in metrics_sum.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_path", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=256)

    # 权重参数
    ap.add_argument("--alpha", type=float, default=1.0, help="Weight for region consistency")
    ap.add_argument("--beta", type=float, default=1.0, help="Weight for category consistency")

    # 论文式 user-conditioned 时空 rerank
    ap.add_argument("--topk_region", type=int, default=40, help="Top-K regions for paper-style rerank")
    ap.add_argument("--topm_time", type=int, default=1, help="Top-M time periods for paper-style rerank")
    ap.add_argument("--gamma", type=float, default=0.0, help="Bias added to POIs that satisfy constraints")
    
    # [新增]
    ap.add_argument("--enable_region_rerank", action="store_false", dest="disable_region_rerank",
                    help="Use this flag to ENABLE region-based rerank. If not set, it defaults to DISABLED.")
    
    # [可选] 显式设置默认值，确保逻辑清晰
    ap.set_defaults(disable_region_rerank=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache_path, map_location="cpu")

    PAD = cache["special_tokens"]["pad"]
    MASK = cache["special_tokens"]["mask"]
    CLS = cache["special_tokens"]["cls"]

    ckpt = load_checkpoint(args.ckpt_path, map_location="cpu")
    cfg_dict = ckpt.hparams.get("cfg", {})

    if not cfg_dict:
        cfg = STBertConfig(
            vocab_size=int(cache["vocab_size"]),
            num_users=int(cache["num_users"]),
            num_cats=int(cache["num_cats"]),
            num_tod=int(cache["num_tod_bins"]) + 1,
            num_dow=8,
            region_vocab_sizes=[int(v) for v in cache["region_vocab_sizes"]],
            max_len=int(cache["max_len"]),
        )
    else:
        cfg = STBertConfig(**cfg_dict)

    model = STBert(
        cfg,
        token2cat=cache["token2cat"],
        token2regions=cache["token2regions"],
        token2xy=cache["token2xy"],
        pad_token=PAD,
        mask_token=MASK,
        cls_token=CLS,
    ).to(device)
    model.load_state_dict(ckpt.model_state, strict=True)

    augment = True if args.split == "train" else False
    ds = NextPOIDataset(
        cache["splits"][args.split],
        token2cat=cache["token2cat"],
        token2regions=cache["token2regions"],
        max_len=cfg.max_len,
        pad_token=PAD,
        cls_token=CLS,
        augment=augment,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_finetune,
        pin_memory=True,
    )

    print(
        f"Evaluating: alpha={args.alpha}, beta={args.beta}, "
        f"gamma={args.gamma}, topK_region={args.topk_region}, topM_time={args.topm_time}, "
        f"disable_region_rerank={args.disable_region_rerank}"
    )
    
    metrics = evaluate(
        model,
        loader,
        device,
        alpha=args.alpha,
        beta=args.beta,
        user_period_pois=cache.get("user_period_pois"),
        user_region_pois=cache.get("user_region_pois"),
        topk_region=args.topk_region,
        topm_time=args.topm_time,
        gamma=args.gamma,
        disable_region_rerank=args.disable_region_rerank, # [传入新参数]
    )
    print(f"{args.split.upper()} : {format_metrics(metrics)}")


if __name__ == "__main__":
    main()