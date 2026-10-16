"""统计指标：封板胜率 + Wilson 区间、显著性检验、收益统计、分层统计。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

RULES = ["r1", "r2", "r3", "r4"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def two_prop_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """两独立比例的 z 检验（双侧），返回 (z, p)。"""
    if min(n1, n2) == 0:
        return (math.nan, math.nan)
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (math.nan, math.nan)
    z = (k1 / n1 - k2 / n2) / se
    return z, 2 * (1 - norm.cdf(abs(z)))


def summarize(ev: pd.DataFrame, rets: pd.DataFrame, idx: pd.Index) -> dict:
    s = ev.loc[idx]
    r = rets.loc[idx]
    n = len(s)
    k = int(s["y_t_close_up"].sum())
    touch = int(s["y_t_touch_up"].sum())
    lo, hi = wilson(k, n)
    out = {
        "样本数": n,
        "交易天数": s["key_t_date"].nunique(),
        "封板率": k / n if n else math.nan,
        "CI下": lo,
        "CI上": hi,
        "触板率": touch / n if n else math.nan,
        "炸板率": 1 - k / touch if touch else math.nan,
        "T日收盘浮盈": s["y_t_ret_close"].mean(),
    }
    for rule in RULES:
        x = r[rule].dropna()
        out[f"{rule}均值"] = x.mean()
        out[f"{rule}中位"] = x.median()
        out[f"{rule}胜率"] = (x > 0).mean() if len(x) else math.nan
    x = r["r1"].dropna()
    pos, neg = x[x > 0], x[x <= 0]
    out["r1盈亏比"] = pos.mean() / abs(neg.mean()) if len(pos) and len(neg) and neg.mean() != 0 else math.nan
    return out


def compare_vs_rest(ev: pd.DataFrame, idx: pd.Index, pool_idx: pd.Index) -> dict:
    """信号组 vs 同一基准池内的其余样本（两组独立），比较封板率。"""
    sig = ev.loc[idx, "y_t_close_up"]
    rest = ev.loc[pool_idx.difference(idx), "y_t_close_up"]
    k1, n1, k2, n2 = int(sig.sum()), len(sig), int(rest.sum()), len(rest)
    z, p = two_prop_test(k1, n1, k2, n2)
    return {
        "信号封板率": k1 / n1 if n1 else math.nan,
        "池内其余封板率": k2 / n2 if n2 else math.nan,
        "提升(百分点)": (k1 / n1 - k2 / n2) * 100 if n1 and n2 else math.nan,
        "z": z,
        "p值": p,
    }


def pnl_structure(ev: pd.DataFrame, rets: pd.DataFrame, idx: pd.Index) -> tuple[pd.DataFrame, float]:
    """按 T 日是否封板拆解收益，并给出 r1 规则下的盈亏平衡封板率。"""
    s, r = ev.loc[idx], rets.loc[idx]
    up = s["y_t_close_up"].astype(bool)
    rows = []
    for name, m in [("封板", up), ("未封板", ~up)]:
        sub = s[m]
        rows.append({
            "分组": name,
            "样本数": int(m.sum()),
            "占比": m.mean() if len(m) else math.nan,
            "T日收盘浮盈": sub["y_t_ret_close"].mean(),
            "T+1开盘溢价": (sub["y_t2_open"] / sub["y_t_close"] - 1).mean(),
            "r1均值": r.loc[m, "r1"].mean(),
            "r1胜率": (r.loc[m, "r1"] > 0).mean(),
            "r2均值": r.loc[m, "r2"].mean(),
            "r3均值": r.loc[m, "r3"].mean(),
        })
    hit, miss = rows[0]["r1均值"], rows[1]["r1均值"]
    breakeven = -miss / (hit - miss) if hit is not None and miss is not None and hit != miss else math.nan
    return pd.DataFrame(rows), breakeven


def by_group(ev: pd.DataFrame, rets: pd.DataFrame, idx: pd.Index, groups: pd.Series) -> pd.DataFrame:
    s = ev.loc[idx]
    r = rets.loc[idx]
    g = groups.loc[idx]
    rows = []
    for name, sub in s.groupby(g, observed=True, sort=True):
        n, k = len(sub), int(sub["y_t_close_up"].sum())
        lo, hi = wilson(k, n)
        rr = r.loc[sub.index]
        rows.append({
            "分组": name,
            "样本数": n,
            "封板率": k / n if n else math.nan,
            "CI下": lo,
            "CI上": hi,
            "触板率": sub["y_t_touch_up"].mean(),
            "r1均值": rr["r1"].mean(),
            "r1胜率": (rr["r1"] > 0).mean(),
            "r3均值": rr["r3"].mean(),
        })
    return pd.DataFrame(rows)
