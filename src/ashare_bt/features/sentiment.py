"""市场情绪日度指标（仅用合格股票池：非 ST、上市满 N 日）。

各指标的可得时点：
- zt/dt/touch/fb/max_height/zb_rate/promo_rate(d)：d 日收盘后
- yzt_auc_norm_mean(d)：d 日 9:25 集合竞价结束后（昨日涨停股今日竞价表现）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_sentiment(panel: pd.DataFrame, elig: pd.Series) -> pd.DataFrame:
    p = panel[elig]
    date = p["date"]
    by = p.groupby(date)
    s = pd.DataFrame({
        "zt_count": by["is_up"].sum(),
        "dt_count": by["is_down"].sum(),
        "touch_count": by["touch_up"].sum(),
        "fb_count": by["is_first_board"].sum(),
        "max_height": by["streak"].max(),
    }).sort_index()
    s["zb_rate"] = np.where(s["touch_count"] > 0, 1 - s["zt_count"] / s["touch_count"], np.nan)

    adjacent = p["prv_cal_i"] == p["cal_i"] - 1
    prv_fb = adjacent & p["prv_is_first_board"].fillna(False).astype(bool)
    prv_up = adjacent & p["prv_is_up"].fillna(False).astype(bool)

    # 首板晋级率：昨日首板中今日涨停的比例（分母为昨日首板总数）
    promo_num = (prv_fb & p["is_up"]).groupby(date).sum()
    s["promo_rate"] = promo_num.reindex(s.index).fillna(0) / s["fb_count"].shift(1)

    # 昨日涨停股今日竞价（开盘）标准化涨幅均值、高开比例
    auc_norm = (p["open"] / p["prev_close"] - 1) / p["limit_pct"]
    s["yzt_auc_norm_mean"] = auc_norm[prv_up].groupby(date[prv_up]).mean()
    s["yzt_gap_up_ratio"] = (auc_norm[prv_up] > 0).groupby(date[prv_up]).mean()
    return s
