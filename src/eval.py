from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
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
    alpha: float = 0.0,
    beta: float = 0.0,
    user_period_pois: dict | None = None,
    user_region_pois: dict | None = None,
    topk_region: int = 40,
    topm_time: int = 1,
    gamma: float = 0.0,
    conf_pow: float = 1.0,
    conf_min: float = 0.0,
    rerank_strategy: str = "combined",  # Default to combined
):
    """
    Evaluate with Hybrid Spatio-Temporal Constraints.
    
    Logic:
    1. Region Score: Boost ALL POIs in the predicted Top-K regions (Vectorized).
    2. Time Score: Boost POIs found in user's history at this time (Set-based).
    """
    model.eval()
    metrics_sum = None
    n = 0

    token2cat = model.token2cat
    # token2region0: Tensor [Vocab_Size], maps POI_ID -> Region_ID
    token2region0 = model._get_token2region(0)

    # Time expansion helper
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

        # Forward pass
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

        # (1) Uncertainty Calculation (Region Confidence)
        # We use this to scale BOTH region and time boosts (optional, but consistent)
        if gamma != 0.0:
            region_logits = reg_logits_list[0]
            B, R_dim = region_logits.shape
            
            # Entropy calculation
            region_log_probs = F.log_softmax(region_logits, dim=-1)
            region_probs = region_log_probs.exp()
            entropy = -(region_probs * region_log_probs).sum(dim=-1)
            
            # Confidence
            if R_dim > 1:
                H_max = float(torch.log(torch.tensor(R_dim, device=entropy.device, dtype=entropy.dtype)))
                entropy_norm = entropy / (H_max + 1e-12)
                entropy_norm = entropy_norm.clamp_(0.0, 1.0)
                confidence = 1.0 - entropy_norm
            else:
                confidence = torch.ones_like(entropy)
                
            if conf_pow != 1.0:
                confidence = confidence.clamp(0.0, 1.0) ** float(conf_pow)
            if conf_min > 0.0:
                confidence = torch.where(confidence >= float(conf_min), confidence, torch.zeros_like(confidence))
        else:
            confidence = None

        # (2) Apply Reranking
        if gamma != 0.0 and confidence is not None:
            
            # --- Part A: Region Weighting (Vectorized - ALL POIs in TopK Region) ---
            # 1. Get Top-K Regions
            K = min(int(topk_region), R_dim)
            region_topk = torch.topk(region_logits, k=K, dim=-1).indices  # [B, K]

            # 2. Create Boost Map
            region_boost_map = torch.zeros(B, R_dim, device=device, dtype=poi_logits.dtype)
            
            # 3. Fill Boost Values (Gamma * Confidence)
            boost_values = confidence.unsqueeze(1).expand(-1, K) * gamma
            region_boost_map.scatter_(1, region_topk, boost_values)
            
            # 4. Project to POI space and Add
            poi_boost_region = region_boost_map[:, token2region0]
            poi_logits = poi_logits + poi_boost_region

            # --- Part B: Time Weighting (History-based - Loop) ---
            # Only if user history is provided
            if user_period_pois is not None:
                # Prepare CPU data for loop
                query_tod = batch["query_tod"].detach().cpu().tolist()
                users_cpu = batch["user"].detach().cpu().tolist()
                
                # Iterate users to apply Time Constraint
                for i, uidx in enumerate(users_cpu):
                    uidx = int(uidx)
                    
                    # Calculate per-sample gamma (same as region)
                    gamma_i = float(gamma) * float(confidence[i].item())
                    if gamma_i < 1e-8:
                        continue

                    # Construct Time-Feasible Set (St)
                    St = set()
                    u_tp = user_period_pois.get(uidx, {})
                    # Look at current time +/- window
                    for t in _time_candidates(int(query_tod[i]), int(topm_time), max_tod):
                        arr = u_tp.get(int(t))
                        if arr is not None and len(arr) > 0:
                            St.update(arr.tolist() if hasattr(arr, "tolist") else list(arr))
                    
                    # Apply Boost for Time
                    if St:
                        idx_t = torch.as_tensor(list(St), device=device, dtype=torch.long)
                        # Add gamma AGAIN for time consistency
                        poi_logits[i, idx_t] = poi_logits[i, idx_t] + gamma_i

        # -----------------------------------------------------------------------

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

    # Uncertainty-Guided rerank params
    ap.add_argument("--topk_region", type=int, default=40, help="Top-K regions")
    ap.add_argument("--topm_time", type=int, default=1, help="Top-M time periods")
    ap.add_argument("--gamma", type=float, default=0.0, help="Base strength for BOTH Region and Time.")
    ap.add_argument("--conf_pow", type=float, default=1.0, help="Confidence sharpening power.")
    ap.add_argument("--conf_min", type=float, default=0.0, help="Min confidence threshold.")

    # Removed alpha/beta/strategy args to keep it clean as requested (assumed 0 internally)
    
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cache
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
        f"Evaluating with Uncertainty-Guided Spatio-Temporal Inference:\n"
        f"  gamma        = {args.gamma}\n"
        f"  topK_region  = {args.topk_region} (Boost ALL POIs in Region)\n"
        f"  topM_time    = {args.topm_time} (Boost History POIs in Time)\n"
        f"  Mechanism    = Region_Boost + Time_Boost"
    )

    metrics = evaluate(
        model,
        loader,
        device,
        alpha=0.0, # Forced 0
        beta=0.0,  # Forced 0
        user_period_pois=cache.get("user_period_pois"),
        user_region_pois=cache.get("user_region_pois"),
        topk_region=args.topk_region,
        topm_time=args.topm_time,
        gamma=args.gamma,
        conf_pow=args.conf_pow,
        conf_min=args.conf_min,
        rerank_strategy="combined", 
    )
    print(f"{args.split.upper()} : {format_metrics(metrics)}")


if __name__ == "__main__":
    main()