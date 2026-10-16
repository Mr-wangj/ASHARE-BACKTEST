"""日频面板：涨停判定、上市天数、连板数、首板、近似流通市值与换手、前后交易日字段。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .limit import add_limit_columns


def _group_shift_cumsum(values: pd.Series, codes: pd.Series, k: int) -> pd.Series:
    """组内：前 k 条记录（不含当日）的求和，用累积和差分实现向量化。"""
    cs = values.astype("int64").groupby(codes).cumsum()
    g = cs.groupby(codes)
    return g.shift(1).fillna(0) - g.shift(k + 1).fillna(0)


def build_panel(
    daily: pd.DataFrame,
    sec: pd.DataFrame,
    float_shares: pd.DataFrame,
    calendar: list[int],
    *,
    first_board_lookback: int = 1,
) -> pd.DataFrame:
    df = daily.merge(sec[["full_code", "name", "board", "is_st_now"]], on="full_code", how="inner")
    df = df.merge(float_shares[["full_code", "float_shares"]], on="full_code", how="left")
    df = df[df["date"].isin(set(calendar))]
    df = df.sort_values(["full_code", "date"]).reset_index(drop=True)
    codes = df["full_code"]
    g = df.groupby(codes, sort=False)

    # 前收：优先用交易所前收（已处理除权），缺失时用上一条收盘价
    df["prev_close"] = df["prev_close"].fillna(g["close"].shift(1))
    df = add_limit_columns(df)

    # 上市天数：首条记录晚于下载区间首个交易日 → 视为区间内上市的新股
    seq = g.cumcount() + 1
    first_date = g["date"].transform("min")
    df["list_days"] = np.where(first_date > calendar[0], seq, 100_000 + seq).astype("int64")

    # 连板数、连续跌停天数（停牌不打断，与行情软件口径一致）
    for flag, col in (("is_up", "streak"), ("is_down", "down_streak")):
        x = df[flag].astype("int64")
        block = ((codes != codes.shift()) | (x == 0)).cumsum()
        df[col] = (x.groupby(block).cumsum() * x).astype("int64")

    # 首板：当日涨停，且前 N 条记录无涨停
    df["up_prev_n"] = _group_shift_cumsum(df["is_up"], codes, first_board_lookback)
    df["is_first_board"] = df["is_up"] & (df["up_prev_n"] == 0)

    # 前 5 日均成交额（不含当日）
    df["amount_ma5_prev"] = (
        g["amount"].shift(1).groupby(codes).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    )

    # 近似流通市值（亿元）与换手率：流通股本为当前快照
    df["float_mv"] = df["close"] * df["float_shares"] / 1e8
    df["turnover_rate"] = df["volume"] / df["float_shares"]

    # 交易日历序号，用于判断前后记录是否为相邻交易日
    cal_index = pd.Series(np.arange(len(calendar)), index=calendar)
    df["cal_i"] = df["date"].map(cal_index).astype("int64")

    # 下一条记录（T 日）与再下一条记录（T+1 卖出日，停牌则顺延）
    nxt_cols = ["date", "cal_i", "open", "high", "low", "close", "prev_close", "up_limit",
                "is_up", "touch_up", "open_up", "amount"]
    for c in nxt_cols:
        df[f"nxt_{c}"] = g[c].shift(-1)
    for c in ["date", "open", "close"]:
        df[f"nx2_{c}"] = g[c].shift(-2)
    # 上一条记录（用于情绪：昨日首板/昨日涨停/昨日跌停）
    for c in ["cal_i", "is_first_board", "is_up", "streak", "is_down", "down_streak"]:
        df[f"prv_{c}"] = g[c].shift(1)
    return df


def eligible_mask(panel: pd.DataFrame, *, exclude_st: bool, min_list_days: int) -> pd.Series:
    m = panel["list_days"] >= min_list_days
    if exclude_st:
        m &= ~panel["is_st_now"]
    return m
