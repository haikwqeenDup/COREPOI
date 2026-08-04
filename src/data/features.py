from __future__ import annotations
import numpy as np
import pandas as pd


def infer_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def parse_utc_datetime(series: pd.Series) -> pd.Series:
    """Parse UTC_time-like column to timezone-aware UTC datetimes."""
    if np.issubdtype(series.dtype, np.number):
        vals = series.astype(np.int64)
        if np.nanmedian(vals) > 10_000_000_000:
            return pd.to_datetime(vals, unit="ms", errors="coerce", utc=True)
        return pd.to_datetime(vals, unit="s", errors="coerce", utc=True)

    s = series.astype(str).str.strip()
    # Common twitter-like format: 'Sun Jun 10 20:07:53 +0000 2012'
    dt = pd.to_datetime(s, format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    # fallback parser
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt


def compute_local_dt(df: pd.DataFrame) -> pd.Series:
    """
    Compute local datetime (timezone-aware) from UTC_time + timezone offset minutes.
    If timezone column missing/invalid, returns UTC dt.
    """
    time_col = infer_col(df, ["UTC_time", "utc_time", "timestamp", "datetime", "time", "created_at"])
    if time_col is None:
        return pd.Series([pd.NaT] * len(df))
    dt_utc = parse_utc_datetime(df[time_col])

    tz_col = infer_col(df, ["timezone", "tz_offset", "utc_offset"])
    if tz_col is None:
        return dt_utc

    tz_min = pd.to_numeric(df[tz_col], errors="coerce")
    # timezone in dataset seems minutes offset from UTC (e.g., -240)
    # local_time = utc_time + offset_minutes
    offset = pd.to_timedelta(tz_min.fillna(0).astype(int), unit="m")
    return (dt_utc + offset)


def tod_bin_from_time_period(time_period: pd.Series, num_tod_bins: int) -> np.ndarray:
    """time_period in [0,1]. Returns bin in [0..num_tod_bins-1], else -1."""
    tp = pd.to_numeric(time_period, errors="coerce")
    out = np.full((len(tp),), -1, dtype=np.int64)
    valid = tp.notna() & (tp >= 0) & (tp <= 1)
    if valid.any():
        bins = np.floor(tp[valid].to_numpy() * float(num_tod_bins)).astype(np.int64)
        bins = np.clip(bins, 0, num_tod_bins - 1)
        out[valid.to_numpy()] = bins
    return out


def tod_bin_from_local_dt(local_dt: pd.Series, num_tod_bins: int) -> np.ndarray:
    """Return TOD bin from local datetime."""
    out = np.full((len(local_dt),), -1, dtype=np.int64)
    valid = local_dt.notna()
    if valid.any():
        h = local_dt.dt.hour[valid].to_numpy()
        m = local_dt.dt.minute[valid].to_numpy()
        minutes = h * 60 + m
        bin_minutes = int((24 * 60) // int(num_tod_bins))
        bins = (minutes // bin_minutes).astype(np.int64)
        bins = np.clip(bins, 0, num_tod_bins - 1)
        out[valid.to_numpy()] = bins
    return out


def dow_from_local_dt(local_dt: pd.Series) -> np.ndarray:
    """Day-of-week in [0..6] Monday=0. Invalid -> -1"""
    out = np.full((len(local_dt),), -1, dtype=np.int64)
    valid = local_dt.notna()
    if valid.any():
        out[valid.to_numpy()] = local_dt.dt.dayofweek[valid].astype(int).to_numpy()
    return out


def latlon_to_xy_meters(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Simple equirectangular projection to meters."""
    R = 6371000.0
    lat = np.deg2rad(lat_deg.astype(np.float64))
    lon = np.deg2rad(lon_deg.astype(np.float64))
    lat0 = np.deg2rad(float(lat0_deg))
    x = R * lon * np.cos(lat0)
    y = R * lat
    return x.astype(np.float32), y.astype(np.float32)


def build_grid_ids(x_m: np.ndarray, y_m: np.ndarray, cell_m: float) -> tuple[np.ndarray, int]:
    gx = np.floor(x_m / float(cell_m)).astype(np.int64)
    gy = np.floor(y_m / float(cell_m)).astype(np.int64)
    mp = {}
    out = np.zeros((len(gx),), dtype=np.int64)
    nid = 0
    for i, key in enumerate(zip(gx.tolist(), gy.tolist())):
        if key not in mp:
            mp[key] = nid + 1  # reserve 0 for unknown
            nid += 1
        out[i] = mp[key]
    return out, nid + 1  # vocab size includes 0
