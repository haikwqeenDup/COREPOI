from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .features import (
    infer_col,
    compute_local_dt,
    tod_bin_from_time_period,
    tod_bin_from_local_dt,
    dow_from_local_dt,
    latlon_to_xy_meters,
    build_grid_ids,
)


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty csv: {path}")
    return df


def _standardize_checkins(df: pd.DataFrame, path: str) -> pd.DataFrame:
    # required
    traj_col = infer_col(df, ["trajectory_id", "traj_id", "trajectory", "seq_id"])
    user_col = infer_col(df, ["user_id", "uid", "user"])
    poi_col = infer_col(df, ["POI_id", "poi_id", "poi", "venue_id"])
    if traj_col is None or user_col is None or poi_col is None:
        raise ValueError(
            f"{path} missing required cols. Need trajectory_id/user_id/POI_id (case-insensitive). Got: {list(df.columns)}"
        )

    # optional
    lat_col = infer_col(df, ["latitude", "lat"])
    lon_col = infer_col(df, ["longitude", "lon", "lng"])
    cat_col = infer_col(df, ["POI_catid", "poi_catid", "catid", "category_id"])
    tp_col = infer_col(df, ["time_period", "tod", "time_bin"])
    tz_col = infer_col(df, ["timezone", "tz_offset", "utc_offset"])
    time_col = infer_col(df, ["UTC_time", "utc_time", "timestamp", "datetime", "time", "created_at"])

    out = pd.DataFrame()
    out["trajectory_id"] = df[traj_col].astype(str)
    out["user_id"] = pd.to_numeric(df[user_col], errors="coerce").astype("Int64")
    out["poi_id"] = df[poi_col].astype(str)

    if lat_col is not None:
        out["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    else:
        out["latitude"] = np.nan

    if lon_col is not None:
        out["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        out["longitude"] = np.nan

    if cat_col is not None:
        out["poi_catid"] = df[cat_col].astype(str)
    else:
        out["poi_catid"] = "NA"

    if tp_col is not None:
        out["time_period"] = pd.to_numeric(df[tp_col], errors="coerce")
    else:
        out["time_period"] = np.nan

    if tz_col is not None:
        out["timezone"] = pd.to_numeric(df[tz_col], errors="coerce")
    else:
        out["timezone"] = np.nan

    if time_col is not None:
        out["UTC_time"] = df[time_col]
    else:
        out["UTC_time"] = None

    # drop rows without essentials
    out = out.dropna(subset=["user_id", "poi_id", "trajectory_id"]).reset_index(drop=True)
    out["user_id"] = out["user_id"].astype(int)

    return out


def _read_poi_info(poi_info_path: str | None) -> pd.DataFrame:
    if poi_info_path is None:
        return pd.DataFrame(columns=["poi_id", "poi_catid", "latitude", "longitude"])
    p = Path(poi_info_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    poi_col = infer_col(df, ["poi_id", "POI_id", "venue_id"])
    lat_col = infer_col(df, ["latitude", "lat"])
    lon_col = infer_col(df, ["longitude", "lon", "lng"])
    cat_col = infer_col(df, ["poi_catid", "POI_catid", "catid", "category_id"])

    if poi_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"{poi_info_path} must contain poi_id, latitude, longitude (case-insensitive).")

    out = pd.DataFrame()
    out["poi_id"] = df[poi_col].astype(str)
    out["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    if cat_col is not None:
        out["poi_catid"] = df[cat_col].astype(str)
    else:
        out["poi_catid"] = "NA"
    out = out.dropna(subset=["poi_id", "latitude", "longitude"]).drop_duplicates("poi_id").reset_index(drop=True)
    return out


def build_cache(
    train_path: str,
    val_path: str,
    test_path: str,
    poi_info_path: str | None,
    cache_path: str,
    max_len: int = 128,
    num_tod_bins: int = 48,
    region_scales_m: List[float] | None = None,
    min_traj_len: int = 2,
    window_stride: int = 64,
) -> None:
    if region_scales_m is None:
        region_scales_m = [700.0, 1200.0, 3000.0]

    # 1) read checkins
    train_df = _standardize_checkins(_read_csv(train_path), train_path)
    val_df = _standardize_checkins(_read_csv(val_path), val_path)
    test_df = _standardize_checkins(_read_csv(test_path), test_path)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 2) read poi_info and merge to obtain complete POI meta
    poi_info_df = _read_poi_info(poi_info_path)

    # POI from checkins (take first non-null lat/lon/catid)
    check_poi = (
        all_df[["poi_id", "poi_catid", "latitude", "longitude"]]
        .dropna(subset=["poi_id"])
        .drop_duplicates("poi_id")
        .reset_index(drop=True)
    )

    poi_info_df = poi_info_df.set_index("poi_id")
    check_poi = check_poi.set_index("poi_id")
    # combine_first keeps existing values in poi_info_df; fill missing from check_poi
    meta = poi_info_df.combine_first(check_poi).reset_index()

    # still missing lat/lon? drop those pois
    meta = meta.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # 3) build POI vocab (poi_idx 0..N-1); token id = poi_idx + 3
    poi_ids = meta["poi_id"].astype(str).tolist()
    poi_id2idx = {pid: i for i, pid in enumerate(poi_ids)}
    idx2poi_id = {i: pid for pid, i in poi_id2idx.items()}
    num_pois = len(poi_ids)

    # 4) users
    user_ids = sorted(set(all_df["user_id"].astype(int).tolist()))
    user_id2idx = {uid: i for i, uid in enumerate(user_ids)}
    idx2user_id = {i: uid for uid, i in user_id2idx.items()}
    num_users = len(user_ids)

    # 5) categories (poi_catid)
    cats_raw = meta["poi_catid"].astype(str).fillna("NA").tolist()
    uniq_cats = sorted(set(cats_raw))
    # reserve 0 for unknown
    cat2idx = {c: i + 1 for i, c in enumerate(uniq_cats)}
    cat2idx["NA"] = 0
    num_cats = len(cat2idx) + 1  # a bit safe
    poi2cat = np.array([cat2idx.get(c, 0) for c in cats_raw], dtype=np.int64)

    # 6) geo xy
    lat = meta["latitude"].astype(float).to_numpy()
    lon = meta["longitude"].astype(float).to_numpy()
    lat0 = float(np.mean(lat))
    x_m, y_m = latlon_to_xy_meters(lat, lon, lat0_deg=lat0)
    # shift to non-negative to stabilize grid id
    x_m = x_m - float(np.min(x_m))
    y_m = y_m - float(np.min(y_m))
    xy_m = np.stack([x_m, y_m], axis=1).astype(np.float32)
    xy_mean = xy_m.mean(axis=0, keepdims=True)
    xy_std = xy_m.std(axis=0, keepdims=True) + 1e-6
    xy_norm = ((xy_m - xy_mean) / xy_std).astype(np.float32)

    # 7) region ids for each scale
    region_scales_m = [float(s) for s in region_scales_m]
    poi2regions = []
    region_vocab_sizes = []
    for cell in region_scales_m:
        rid, vocab = build_grid_ids(x_m, y_m, cell_m=cell)
        poi2regions.append(rid.astype(np.int64))
        region_vocab_sizes.append(int(vocab))

    # 8) build token-level meta arrays (length vocab_size)
    PAD, MASK, CLS = 0, 1, 2
    OFFSET = 3
    vocab_size = num_pois + OFFSET

    token2cat = np.zeros((vocab_size,), dtype=np.int64)
    token2cat[OFFSET:] = poi2cat

    token2xy = np.zeros((vocab_size, 2), dtype=np.float32)
    token2xy[OFFSET:, :] = xy_norm

    token2xy_m = np.zeros((vocab_size, 2), dtype=np.float32)
    token2xy_m[OFFSET:, :] = xy_m

    token2regions = []
    for rid in poi2regions:
        arr = np.zeros((vocab_size,), dtype=np.int64)
        arr[OFFSET:] = rid
        token2regions.append(arr)

    # 9) build trajectories
    def build_trajs(split_df: pd.DataFrame, split_name: str) -> List[Tuple[int, List[int], List[int], List[int]]]:
        # compute time features aligned to rows first, then slice per traj
        local_dt = compute_local_dt(split_df)

        # TOD: prefer time_period if present and valid
        if "time_period" in split_df.columns and split_df["time_period"].notna().any():
            tod = tod_bin_from_time_period(split_df["time_period"], num_tod_bins=num_tod_bins)
            missing = (tod == -1)
            if missing.any():
                tod2 = tod_bin_from_local_dt(local_dt, num_tod_bins=num_tod_bins)
                tod[missing] = tod2[missing]
        else:
            tod = tod_bin_from_local_dt(local_dt, num_tod_bins=num_tod_bins)

        dow = dow_from_local_dt(local_dt)

        # sorting key
        dt_sort = local_dt
        has_dt = dt_sort.notna().any()

        trajs: List[Tuple[int, List[int], List[int], List[int]]] = []
        for traj_id, g in split_df.groupby("trajectory_id"):
            if has_dt:
                # sort by local dt; NaT will go last
                g = g.assign(_dt=dt_sort.loc[g.index].values).sort_values("_dt")
            u = int(g["user_id"].iloc[0])
            if u not in user_id2idx:
                continue
            uidx = user_id2idx[u]

            poi_ids_g = g["poi_id"].astype(str).tolist()
            idxs = [poi_id2idx.get(pid, -1) for pid in poi_ids_g]
            # filter invalid and align time features using boolean mask on original g order
            keep = np.array([i >= 0 for i in idxs], dtype=bool)
            if keep.sum() < min_traj_len:
                continue
            idxs = np.array([i for i in idxs if i >= 0], dtype=np.int64)
            # map to token ids
            tokens = (idxs + OFFSET).tolist()

            # align time features
            tod_g = tod[g.index.to_numpy()]
            dow_g = dow[g.index.to_numpy()]
            tod_keep = tod_g[keep].astype(np.int64).tolist()
            dow_keep = dow_g[keep].astype(np.int64).tolist()

            # clip if too long? keep full for now; dataset will window/crop
            trajs.append((uidx, tokens, tod_keep, dow_keep))
        return trajs

    train_trajs = build_trajs(train_df, "train")
    val_trajs = build_trajs(val_df, "val")
    test_trajs = build_trajs(test_df, "test")

    
    # 10) user-conditioned mappings for rerank (built from TRAIN only)
    # Following the idea: keep only POIs the user has visited under certain region/time conditions.
    # We store as sparse dict-of-dicts of unique POI tokens to avoid huge dense tensors.
    #
    # user_period_pois[u][t] -> np.ndarray of POI token ids the user visited in time period t
    #   - key t is query_tod space (tod_label + 1), so valid range: 1..num_tod_bins, 0 means unknown
    # user_region_pois[u][r] -> np.ndarray of POI token ids the user visited in region r (scale 0)
    #   - region ids already reserve 0 for unknown (see build_grid_ids)
    from collections import defaultdict

    user_period_sets = defaultdict(lambda: defaultdict(set))
    user_region_sets = defaultdict(lambda: defaultdict(set))
    token2region0_np = token2regions[0]  # np.ndarray [V] -> region_id at finest scale

    for uidx, toks, tod_list, _ in train_trajs:
        # toks are token ids (>=3 for POIs)
        for tok, tod_label in zip(toks, tod_list):
            if tok < OFFSET:
                continue
            # time period key in the same space as dataset's query_tod (label + 1)
            tkey = int(tod_label) + 1 if int(tod_label) >= 0 else 0
            if tkey != 0:
                user_period_sets[int(uidx)][tkey].add(int(tok))

            rkey = int(token2region0_np[int(tok)])
            if rkey != 0:
                user_region_sets[int(uidx)][rkey].add(int(tok))

    # convert sets to compact numpy arrays for serialization
    user_period_pois = {
        int(u): {int(t): np.fromiter(sorted(s), dtype=np.int64) for t, s in td.items()}
        for u, td in user_period_sets.items()
    }
    user_region_pois = {
        int(u): {int(r): np.fromiter(sorted(s), dtype=np.int64) for r, s in rd.items()}
        for u, rd in user_region_sets.items()
    }

# 10) popularity from train
    pop = np.zeros((vocab_size,), dtype=np.float32)
    for _, toks, _, _ in train_trajs:
        for t in toks:
            pop[t] += 1.0
    # avoid log(0)
    pop = np.log1p(pop)

    cache = {
        "special_tokens": {"pad": PAD, "mask": MASK, "cls": CLS, "offset": OFFSET},
        "max_len": int(max_len),
        "num_tod_bins": int(num_tod_bins),
        "region_scales_m": region_scales_m,
        "region_vocab_sizes": region_vocab_sizes,
        "vocab_size": int(vocab_size),
        "num_pois": int(num_pois),
        "num_users": int(num_users),
        "num_cats": int(max(token2cat.max() + 1, 1)),
        "poi_id2idx": poi_id2idx,
        "idx2poi_id": idx2poi_id,
        "user_id2idx": user_id2idx,
        "idx2user_id": idx2user_id,
        "cat2idx": cat2idx,
        "token2cat": token2cat,
        "token2xy": token2xy,
        "token2xy_m": token2xy_m,
        "token2regions": token2regions,
        "token_pop": pop,
        "user_period_pois": user_period_pois,
        "user_region_pois": user_region_pois,
        "splits": {
            "train": train_trajs,
            "val": val_trajs,
            "test": test_trajs,
        },
        "meta": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "poi_info_path": str(poi_info_path) if poi_info_path is not None else None,
        },
    }

    cache_path = str(cache_path)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)

    print(f"Saved cache to: {cache_path}")
    print(f"POIs: {num_pois}, Users: {num_users}, Vocab: {vocab_size}")
    print(f"Train trajs: {len(train_trajs)}, Val: {len(val_trajs)}, Test: {len(test_trajs)}")
    print(f"TOD bins: {num_tod_bins}, Region scales: {region_scales_m} (vocab sizes={region_vocab_sizes})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, required=True)
    ap.add_argument("--val_path", type=str, required=True)
    ap.add_argument("--test_path", type=str, required=True)
    ap.add_argument("--poi_info_path", type=str, default=None)
    ap.add_argument("--cache_path", type=str, required=True)

    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--num_tod_bins", type=int, default=48)
    ap.add_argument("--region_scales_m", type=str, default="700,1200,3000")
    ap.add_argument("--min_traj_len", type=int, default=2)
    ap.add_argument("--window_stride", type=int, default=64)

    args = ap.parse_args()
    region_scales = [float(x) for x in args.region_scales_m.split(",") if x.strip()]
    build_cache(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        poi_info_path=args.poi_info_path,
        cache_path=args.cache_path,
        max_len=args.max_len,
        num_tod_bins=args.num_tod_bins,
        region_scales_m=region_scales,
        min_traj_len=args.min_traj_len,
        window_stride=args.window_stride,
    )


if __name__ == "__main__":
    main()
