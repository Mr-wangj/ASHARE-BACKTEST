"""组合资金曲线。

资金分 A/B 两份按交易日奇偶轮换：T 日竞价买入、T+1 卖出，竞价卖出的资金赶不上同一场竞价买入，
所以同一份资金最快隔日复用。每份内当日信号等权。T+1 停牌顺延导致的资金占用延长未建模。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 242


def equity_curve(ev: pd.DataFrame, rets: pd.DataFrame, idx: pd.Index, rule: str, dates: list[int]) -> pd.DataFrame:
    s = ev.loc[idx]
    day_ret = rets.loc[idx, rule].groupby(s["key_t_date"]).mean()
    navs = [1.0, 1.0]
    rows = []
    for i, d in enumerate(dates):
        x = day_ret.get(d, np.nan)
        has = not pd.isna(x)
        navs[i % 2] *= 1 + (x if has else 0.0)
        rows.append((d, (navs[0] + navs[1]) / 2, x if has else 0.0, has))
    return pd.DataFrame(rows, columns=["date", "nav", "day_ret", "has_signal"])


def curve_stats(curve: pd.DataFrame) -> dict:
    nav = curve["nav"]
    if nav.empty:
        return {}
    ret = nav.pct_change().fillna(nav.iloc[0] - 1)
    n = len(nav)
    total = nav.iloc[-1] - 1
    ann = (nav.iloc[-1]) ** (TRADING_DAYS / n) - 1 if nav.iloc[-1] > 0 else math.nan
    dd = (nav / nav.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * math.sqrt(TRADING_DAYS) if ret.std() > 0 else math.nan
    return {
        "累计收益": total,
        "年化收益": ann,
        "最大回撤": dd,
        "夏普": sharpe,
        "有信号天数": int(curve["has_signal"].sum()),
        "总天数": n,
    }
