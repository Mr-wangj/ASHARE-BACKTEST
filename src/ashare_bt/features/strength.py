"""T-1 首板涨停强度排序。

- time（默认）：按 T-1 首次涨停时间排序，越早越强（同一时间按板型 一字>T字>实体、T-1 竞价涨幅）。
  首次涨停时间来自同花顺问财，精确到秒；即“每日一进二明细”页的 T-1日序号。
- lex / weighted：早期的日线代理版，用于对比。
  thsdk 无两年历史分时时，用日线指标近似“是否炸板 / 封板早晚”：
  板型（一字 > T字 > 实体）作为第一排序键；同板型内用代理分 = 以下三项截面分位的均值（越大越强）：
     - 开盘涨幅（标准化）：高开越多，越倾向早封板
     - 换手率（低者强）：早封板者封后成交萎缩；尾盘板、烂板全天换手大
     - 盘中最大回撤 (涨停价-最低价)/前收（小者强）：炸板回落、先抑后扬都会放大该值
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRENGTH_MODES = {
    "time": "按首次涨停时间（越早越强）",
    "lex": "字典序：板型 → 日线代理分",
    "weighted": "加权：板型 40% + 代理分 60%",
}


def add_strength(ev: pd.DataFrame, mode: str = "time") -> pd.DataFrame:
    """写入 x_proxy、x_str_rank（1 = 最强）、x_P_str（0 最强，1 最弱）。time 模式需要先有 x_t1_rank。"""
    key = ev["key_t_date"]
    r_open = ev["x_t1_open_norm"].groupby(key).rank(pct=True)
    r_turn = (-ev["x_t1_turnover"]).groupby(key).rank(pct=True)
    r_dd = (-ev["x_t1_dd"]).groupby(key).rank(pct=True)
    ev["x_proxy"] = pd.concat([r_open, r_turn, r_dd], axis=1).mean(axis=1, skipna=True).fillna(0.5)

    if mode == "time":
        if "x_t1_rank" not in ev.columns:
            raise ValueError("time 模式需要 x_t1_rank，请先计算当日序号（add_day_ranks）")
        # 缺少首次涨停时间的（如代码切换前的北交所）在序号里排最后，但不参与强弱分位，避免被误判为弱板
        valid = ev["x_t1_first_sec"].notna()
        n_valid = valid.groupby(key).transform("sum")
        rank = ev["x_t1_rank"].astype("float64")
        ev["x_str_rank"] = rank
        ev["x_P_str"] = np.where(valid, np.where(n_valid > 1, (rank - 1) / (n_valid - 1).clip(lower=1), 0.5), np.nan)
        return ev
    if mode in ("lex", "weighted"):
        if mode == "lex":
            score = ev["x_bt"].astype("float64") * 10 + ev["x_proxy"]
        else:
            score = 0.4 * (ev["x_bt"].astype("float64") - 1) / 2 + 0.6 * ev["x_proxy"]
        rank = score.groupby(key).rank(ascending=False, method="average")
    else:
        raise ValueError(f"未知 strength_mode: {mode}")

    n = key.map(key.value_counts())
    ev["x_str_rank"] = rank
    ev["x_P_str"] = np.where(n > 1, (rank - 1) / (n - 1).clip(lower=1), 0.5)
    return ev
