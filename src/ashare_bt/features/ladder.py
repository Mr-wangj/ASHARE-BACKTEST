"""每日涨停天梯（build_ladder）与跌停天梯（build_down_ladder），供看板「涨停天梯」「跌停天梯」页使用。

涨停天梯：当日涨停的股票按连板高度分层，并附上冲击该层失败的股票。

每行 = 某日的一只股票，kind 三种：
- 涨停：当日封住涨停，所在层 row = 连板数 streak；
- 炸板：当日触及涨停价但没封住，所在层 row = 昨日连板数 + 1（首板炸板为第 1 层）；
- 断板：昨日涨停、当日未触及涨停价，所在层 row = 昨日连板数 + 1。
股票池与每日明细一致（剔除 ST、上市不足 N 个交易日）；连板数沿用面板口径（停牌不打断）。
晋级率：昨日 h-1 板中，今日（相邻交易日）涨停成为 h 板的比例。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import eligible_mask

KIND_UP, KIND_ZB, KIND_DK = "涨停", "炸板", "断板"
KIND_DT, KIND_QB, KIND_ZD = "跌停", "翘板", "止跌"   # 跌停天梯：封住跌停 / 触及跌停未封住 / 昨日跌停、今日未触及
DOWN_COLS = ["date", "code", "name", "board", "kind", "is_down", "row", "down_streak", "prv_down_streak", "prv_adj",
             "prv_down", "prv_up", "bt", "open_pct", "close_pct", "amount", "turnover", "float_mv",
             "max_streak", "dd_high", "tiandi", "ditian", "ths_warp"]
LADDER_COLS = ["date", "code", "name", "board", "kind", "is_up", "row", "streak", "prv_streak", "prv_adj", "prv_up",
               "bt", "open_pct", "close_pct", "amount", "turnover", "float_mv",
               "first_time", "first_sec", "reason", "auc_amt", "prv_reason"]


def build_ladder(panel: pd.DataFrame, calendar: list[int], cfg: dict, *, limitup: pd.DataFrame | None = None,
                 start: int | None = None) -> pd.DataFrame:
    """streak：当日连板数（未涨停为 0）；prv_streak：上一条记录的连板数；prv_adj：上一条记录是前一交易日（没有停牌）；
    prv_up：昨日（相邻交易日）涨停。limitup：问财涨停明细，合并当日的首次涨停时间、涨停原因、9:25 竞价成交额，
    以及昨日的涨停原因 prv_reason（炸板、断板股当天没有涨停原因，用昨日的归题材）。"""
    u = cfg["universe"]
    elig = eligible_mask(panel, exclude_st=u["exclude_st"], min_list_days=u["min_list_days"])
    prv_adj = panel["prv_cal_i"] == panel["cal_i"] - 1
    prv_up = prv_adj & panel["prv_is_up"].fillna(False).astype(bool)
    touch = panel["touch_up"].astype(bool)
    m = elig & (touch | prv_up)
    if start is not None:
        m &= panel["date"] >= start
    p = panel.loc[m]
    is_up = p["is_up"].astype(bool).to_numpy()
    streak = p["streak"].astype("int64").to_numpy()
    prv_streak = p["prv_streak"].fillna(0).astype("int64").to_numpy()
    cal = np.asarray(calendar, dtype="int64")
    adj = prv_adj[m].to_numpy()
    out = pd.DataFrame({
        "date": p["date"].astype("int64").to_numpy(),
        "code": p["full_code"].to_numpy(),
        "name": p["name"].to_numpy(),
        "board": p["board"].to_numpy(),
        "kind": np.select([is_up, touch[m].to_numpy()], [KIND_UP, KIND_ZB], default=KIND_DK),
        "is_up": is_up,
        "row": np.where(is_up, streak, prv_streak + 1),
        "streak": streak,
        "prv_streak": prv_streak,
        "prv_adj": adj,
        "prv_up": prv_up[m].to_numpy(),
        "bt": p["board_type"].astype("int64").to_numpy(),
        "open_pct": (p["open"] / p["prev_close"] - 1).to_numpy(),
        "close_pct": (p["close"] / p["prev_close"] - 1).to_numpy(),
        "amount": p["amount"].astype("float64").to_numpy(),
        "turnover": p["turnover_rate"].astype("float64").to_numpy(),
        "float_mv": p["float_mv"].astype("float64").to_numpy(),
        "prv_date": np.where(adj, cal[np.clip(p["cal_i"].to_numpy() - 1, 0, None)], 0),
    })
    out["c6"] = out["code"].str[4:]
    if limitup is not None and not limitup.empty:
        lu = limitup.rename(columns={"code": "c6", "auc_amt_t1": "auc_amt"})
        out = out.merge(lu[["c6", "t1_date", "first_time", "first_sec", "reason", "auc_amt"]]
                        .rename(columns={"t1_date": "date"}), on=["c6", "date"], how="left")
        out = out.merge(lu[["c6", "t1_date", "reason"]].rename(columns={"t1_date": "prv_date", "reason": "prv_reason"}),
                        on=["c6", "prv_date"], how="left")
    else:
        for c in ["first_time", "reason", "prv_reason"]:
            out[c] = pd.NA
        out["first_sec"], out["auc_amt"] = np.nan, np.nan
    for c in ["first_time", "reason", "prv_reason"]:
        out[c] = out[c].astype("string")
    for c in ["first_sec", "auc_amt"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out[LADDER_COLS].sort_values(["date", "row", "is_up", "first_sec"], ascending=[True, False, False, True],
                                        na_position="last").reset_index(drop=True)


def build_down_ladder(panel: pd.DataFrame, cfg: dict, *, start: int | None = None, lookback: int = 20) -> pd.DataFrame:
    """跌停天梯：每行 = 某日的一只股票，kind 三种：
    - 跌停：收盘价等于跌停价，row = 连续跌停天数 down_streak（停牌不打断）；
    - 翘板：盘中最低价触及跌停价、收盘没封住（含地天板），row = 昨日连续跌停天数 + 1；
    - 止跌：昨日跌停、今日没触及跌停价，row = 昨日连续跌停天数 + 1。
    附加：max_streak = 近 lookback 条交易记录（含当日）的最高连板数（≥3 即高位股补跌），dd_high = 收盘价距其中
    最高收盘价的回撤，tiandi = 天地板（盘中触及涨停、收盘跌停），ditian = 地天板（盘中触及跌停、收盘涨停），
    prv_up = 昨日涨停（今日跌停即“昨涨今跌”），ths_warp = 同花顺记为翘板（收盘虽在跌停价，同花顺判定未封住的也记翘板）。
    股票池与涨停天梯一致。"""
    u = cfg["universe"]
    elig = eligible_mask(panel, exclude_st=u["exclude_st"], min_list_days=u["min_list_days"])
    prv_adj = panel["prv_cal_i"] == panel["cal_i"] - 1
    prv_down = prv_adj & panel["prv_is_down"].fillna(False).astype(bool)
    touch = panel["touch_down"].astype(bool)
    m = elig & (touch | prv_down)
    if start is not None:
        m &= panel["date"] >= start
    g = panel.groupby("full_code", sort=False)
    max_streak = g["streak"].rolling(lookback, min_periods=1).max().reset_index(level=0, drop=True)
    high = g["close"].rolling(lookback, min_periods=1).max().reset_index(level=0, drop=True)
    p = panel.loc[m]
    is_dn = p["is_down"].astype(bool).to_numpy()
    out = pd.DataFrame({
        "date": p["date"].astype("int64").to_numpy(),
        "code": p["full_code"].to_numpy(),
        "name": p["name"].to_numpy(),
        "board": p["board"].to_numpy(),
        "kind": np.select([is_dn, touch[m].to_numpy()], [KIND_DT, KIND_QB], default=KIND_ZD),
        "is_down": is_dn,
        "row": np.where(is_dn, p["down_streak"].to_numpy(), p["prv_down_streak"].fillna(0).astype("int64").to_numpy() + 1),
        "down_streak": p["down_streak"].astype("int64").to_numpy(),
        "prv_down_streak": p["prv_down_streak"].fillna(0).astype("int64").to_numpy(),
        "prv_adj": prv_adj[m].to_numpy(),
        "prv_down": prv_down[m].to_numpy(),
        "prv_up": (prv_adj & panel["prv_is_up"].fillna(False).astype(bool))[m].to_numpy(),
        "bt": p["board_type_down"].astype("int64").to_numpy(),
        "open_pct": (p["open"] / p["prev_close"] - 1).to_numpy(),
        "close_pct": (p["close"] / p["prev_close"] - 1).to_numpy(),
        "amount": p["amount"].astype("float64").to_numpy(),
        "turnover": p["turnover_rate"].astype("float64").to_numpy(),
        "float_mv": p["float_mv"].astype("float64").to_numpy(),
        "max_streak": max_streak.loc[p.index].astype("int64").to_numpy(),
        "dd_high": (p["close"] / high.loc[p.index] - 1).to_numpy(),
        "tiandi": (p["is_down"] & p["touch_up"]).astype(bool).to_numpy(),
        "ditian": (p["is_up"] & p["touch_down"]).astype(bool).to_numpy(),
        "ths_warp": p["ths_warp"].fillna(False).astype(bool).to_numpy(),
    })
    return out[DOWN_COLS].sort_values(["date", "row", "is_down", "bt"], ascending=[True, False, False, False]
                                      ).reset_index(drop=True)


def ladder_levels(ladder: pd.DataFrame, day: int, prev_day: int | None, *, flag: str = "is_up",
                  streak: str = "streak") -> pd.DataFrame:
    """某日天梯的各层（从高到低）：height、n（当日涨停只数）、fail（冲击该层失败：炸板 + 断板）、
    promo_num / promo_den（晋级：今日 h 板中昨日为 h-1 板的只数 / 昨日 h-1 板只数；首板为空）。
    跌停天梯传 flag="is_down", streak="down_streak"：n 为连续跌停 h 天的只数，promo 为续跌（昨日 h-1 连跌 → 今日 h）。"""
    today = ladder[ladder["date"] == day]
    up = today[today[flag]]
    n = up.groupby(streak).size()
    num = up[up["prv_adj"]].groupby(streak).size()
    fail = today[~today[flag]].groupby("row").size()
    yday = (ladder[(ladder["date"] == prev_day) & ladder[flag]].groupby(streak).size()
            if prev_day is not None else pd.Series(dtype="int64"))
    heights = set(n.index) | set(fail.index) | {h + 1 for h in yday.index}
    rows = []
    for h in sorted(heights, reverse=True):
        promo = h >= 2 and prev_day is not None
        rows.append({"height": int(h), "n": int(n.get(h, 0)), "fail": int(fail.get(h, 0)),
                     "promo_num": int(num.get(h, 0)) if promo else pd.NA,
                     "promo_den": int(yday.get(h - 1, 0)) if promo else pd.NA})
    out = pd.DataFrame(rows, columns=["height", "n", "fail", "promo_num", "promo_den"])
    return out.astype({"promo_num": "Int64", "promo_den": "Int64"})
