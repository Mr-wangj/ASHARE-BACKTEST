"""事件宽表：每行 = 一个（T 日, 股票）候选，T-1 为该股首板日（level=1，回测用）；
每日明细另有二板（level=2，二进三）、三板（level=3，三进四）。

列前缀约定（从机制上防止前视）：
- key_：标识列（日期、代码）
- x_  ：信号可用特征（T-1 收盘后 + T 日 9:25 竞价可得）
- y_  ：结果列（T 日盘中/收盘、T+1），只用于评估，策略函数不可读取
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.costs import net_return
from .panel import eligible_mask
from .strength import add_strength

SENT_T1_COLS = ["zt_count", "dt_count", "zb_rate", "fb_count", "max_height", "promo_rate"]
DETAIL_COLS = ["first_time", "first_sec", "reason", "auc_amt_t1", "auc_amt_t"]
PREVIEW_T_DATE = 99999999  # 次日候选（最新交易日的首板）的 T 日占位：下一交易日尚未到来
DETAIL_LEVELS = (1, 2, 3)  # 每日明细的板位：1 一进二（T-1 首板）、2 二进三（二板）、3 三进四（三板）
_CN = "零一二三四五六七八九十"


def board_name(level: int) -> str:
    """T-1 的板位：首板、二板、三板……"""
    return "首板" if level == 1 else f"{_CN[level]}板"


def level_name(level: int) -> str:
    """晋级名：一进二、二进三、三进四……"""
    return f"{_CN[level]}进{_CN[level + 1]}"


def board_mask(panel: pd.DataFrame, level: int) -> pd.Series:
    """当日为第 level 板：连续 level 个交易日涨停（停牌不打断），且这串连板的第一天是首板。
    首板口径见 first_board_lookback；为 1 时二板 = 首板次日再封板，即前一天一进二的封板股。"""
    if level == 1:
        return panel["is_first_board"]
    start_is_first = panel.groupby("full_code", sort=False)["is_first_board"].shift(level - 1, fill_value=False)
    return (panel["streak"] == level) & start_is_first.astype(bool)


def _seq_rank(ev: pd.DataFrame, by: list[str], ascending: list[bool]) -> pd.Series:
    """同一 T 日内按 by 排序后的序号（1 起，严格递增；缺失值排在最后）。"""
    order = ev.sort_values(["key_t_date", *by], ascending=[True, *ascending], na_position="last", kind="mergesort")
    return order.groupby("key_t_date", sort=False).cumcount().add(1).reindex(ev.index).astype("int64")


def add_day_ranks(ev: pd.DataFrame) -> pd.DataFrame:
    """T-1日序号：T-1 首次涨停时间越早越靠前（同一时间按板型 一字>T字>实体、T-1 竞价涨幅、代码）；
    T日序号：T 日竞价涨幅从高到低（相同则竞价成交额大者在前）。序号基于当日全部首板候选。"""
    ev["x_t1_rank"] = _seq_rank(ev, ["x_t1_first_sec", "x_bt", "x_t1_auc_pct", "key_code"],
                                [True, False, False, True])
    ev["x_t_rank"] = _seq_rank(ev, ["x_auc_pct", "x_t_auc_amt", "key_code"], [False, False, True])
    return ev


def add_cross_section_ranks(ev: pd.DataFrame, strength_mode: str) -> pd.DataFrame:
    """同一 T 日首板池内：当日序号、强度分位 P_str、竞价分位 P_auc（0 为最高）、排名跃迁 Δ。"""
    key = ev["key_t_date"]
    ev["x_pool_n"] = key.map(key.value_counts()).astype("int64")
    add_day_ranks(ev)
    add_strength(ev, strength_mode)
    r_auc = ev["x_auc_norm"].groupby(key).rank(ascending=False, method="average")
    ev["x_P_auc"] = np.where(ev["x_pool_n"] > 1, (r_auc - 1) / (ev["x_pool_n"] - 1).clip(lower=1), 0.5)
    ev["x_delta"] = ev["x_P_str"] - ev["x_P_auc"]
    return ev


def build_events(
    panel: pd.DataFrame,
    sent: pd.DataFrame,
    calendar: list[int],
    cfg: dict,
    *,
    limitup: pd.DataFrame | None = None,
    include_last_day: bool = False,
    include_preview: bool = False,
    level: int = 1,
) -> pd.DataFrame:
    """limitup：问财涨停明细（code, t1_date, first_time, first_sec, reason, auc_amt_t1, auc_amt_t）。
    include_last_day：包含最后一个交易日作为 T 日（此时没有 T+1 结果，只用于每日明细页，不用于回测）。
    include_preview：再加上最后一个交易日的首板（次日一进二候选）。这些行 key_t_date = PREVIEW_T_DATE，
    key_preview = True，T 日及之后的字段全部为空，只用于每日明细页。
    level：T-1 的板位（见 board_mask），1 = 首板（一进二，回测口径）、2 = 二板（二进三）、3 = 三板（三进四）；
    排序、分位都在同一板位、同一 T 日的候选池内计算。"""
    u, b = cfg["universe"], cfg["backtest"]
    elig = eligible_mask(panel, exclude_st=u["exclude_st"], min_list_days=u["min_list_days"])
    t_adjacent = panel["nxt_cal_i"] == panel["cal_i"] + 1  # T 日为 T-1 的下一个交易日且未停牌
    on_board = board_mask(panel, level)
    cand = panel[elig & on_board & t_adjacent].copy()

    end = calendar[-1] if include_last_day else (b.get("end") or calendar[-2])  # 回测保证 T+1 有数据
    t_date = cand["nxt_date"].astype("int64")
    cand = cand[(t_date >= b["start"]) & (t_date <= end)]
    if include_preview:  # 最新交易日的首板（或二板、三板）：还没有 T 日
        prev = panel[elig & on_board & (panel["date"] == calendar[-1])]
        cand = pd.concat([cand, prev])
    is_prev = cand["nxt_date"].isna()

    # ---- 合并问财涨停明细（首次涨停时间、涨停原因、T-1/T 日 9:25 竞价成交额）
    cand["code"] = cand["full_code"].str[4:]
    if limitup is not None and not limitup.empty:
        lu = limitup.rename(columns={"t1_date": "date"})[["code", "date", *DETAIL_COLS]]
        cand = cand.merge(lu, on=["code", "date"], how="left").set_index(cand.index)
    else:
        for c in DETAIL_COLS:
            cand[c] = np.nan

    ev = pd.DataFrame(index=cand.index)
    ev["key_code"] = cand["full_code"]
    ev["key_name"] = cand["name"]
    ev["key_t1_date"] = cand["date"].astype("int64")
    ev["key_t_date"] = cand["nxt_date"].fillna(PREVIEW_T_DATE).astype("int64")
    ev["key_preview"] = is_prev.to_numpy()
    ev["key_level"] = np.int64(level)

    # ---- T-1 特征（收盘后可得）
    ev["x_board"] = cand["board"]
    ev["x_limit_pct"] = cand["limit_pct"]
    ev["x_bt"] = cand["board_type"].astype("int64")
    ev["x_t1_open_norm"] = (cand["open"] / cand["prev_close"] - 1) / cand["limit_pct"]
    ev["x_t1_dd"] = (cand["up_limit"] - cand["low"]) / cand["prev_close"] / cand["limit_pct"]
    ev["x_t1_turnover"] = cand["turnover_rate"]
    ev["x_t1_amount"] = cand["amount"]
    ev["x_t1_float_mv"] = cand["float_mv"]
    ev["x_t1_vol_ratio"] = cand["amount"] / cand["amount_ma5_prev"]
    ev["x_t1_ths_up"] = cand["ths_up"].astype(bool)
    ev["x_t1_auc_pct"] = cand["open"] / cand["prev_close"] - 1
    ev["x_t1_first_time"] = cand["first_time"].astype("string")
    ev["x_t1_first_sec"] = pd.to_numeric(cand["first_sec"], errors="coerce").astype("float64")
    ev["x_t1_reason"] = cand["reason"].astype("string")
    ev["x_t1_auc_amt"] = pd.to_numeric(cand["auc_amt_t1"], errors="coerce").astype("float64")

    # ---- T 日竞价特征（开盘价 = 9:25 集合竞价成交价；成交额来自问财 9:25 分时成交额）
    t_open = cand["nxt_open"].astype("float64")
    ev["x_auc_pct"] = t_open / cand["nxt_prev_close"].astype("float64") - 1
    ev["x_auc_norm"] = ev["x_auc_pct"] / cand["limit_pct"]
    ev["x_open_limit"] = cand["nxt_open_up"].fillna(False).astype(bool)
    ev["x_t_auc_amt"] = pd.to_numeric(cand["auc_amt_t"], errors="coerce").astype("float64")

    # ---- 截面排序（同一 T 日的首板池）
    add_cross_section_ranks(ev, cfg["strategy"]["strength_mode"])
    if is_prev.any():  # 次日候选还没有 T 日竞价：T日序号留空
        ev["x_t_rank"] = ev["x_t_rank"].astype("float64").where(~ev["key_preview"])

    # ---- 情绪：T-1 收盘后指标 + T 日竞价指标
    s_t1 = sent[SENT_T1_COLS].add_prefix("x_s_").add_suffix("_t1")
    ev = ev.join(s_t1, on="key_t1_date")
    ev["x_s_yzt_auc_T"] = ev["key_t_date"].map(sent["yzt_auc_norm_mean"])
    ev["x_s_yzt_gapup_T"] = ev["key_t_date"].map(sent["yzt_gap_up_ratio"])

    # ---- 结果（仅评估）
    ev["y_t_open"] = t_open
    ev["y_t_close"] = cand["nxt_close"].astype("float64")
    ev["y_t_close_up"] = cand["nxt_is_up"].fillna(False).astype(bool)
    ev["y_t_touch_up"] = cand["nxt_touch_up"].fillna(False).astype(bool)
    ev["y_t_ret_close"] = ev["y_t_close"] / t_open - 1
    ev["y_t2_date"] = cand["nx2_date"]
    ev["y_t2_open"] = cand["nx2_open"].astype("float64")
    ev["y_t2_close"] = cand["nx2_close"].astype("float64")
    return ev.reset_index(drop=True)


RETURN_RULES = {
    "r1": "T+1开盘卖",
    "r2": "T+1收盘卖",
    "r3": "封板者T+1收盘卖/否则开盘卖",
    "r4": "T+1低开>3%开盘止损/否则收盘卖",
}


def compute_returns(ev: pd.DataFrame, costs: dict) -> pd.DataFrame:
    """按卖出规则计算扣费净收益；T+1 停牌已在面板中顺延到下一条记录。"""
    buy = ev["y_t_open"]
    o2, c2 = ev["y_t2_open"], ev["y_t2_close"]
    out = pd.DataFrame(index=ev.index)
    out["r1"] = net_return(buy, o2, costs)
    out["r2"] = net_return(buy, c2, costs)
    out["r3"] = net_return(buy, np.where(ev["y_t_close_up"], c2, o2), costs)
    gap = o2 / ev["y_t_close"] - 1
    out["r4"] = net_return(buy, np.where(gap < -0.03, o2, c2), costs)
    return out
