"""涨跌停价计算与涨停/板型判定。

规则（已用同花顺涨跌停事件逐条核对）：
- 沪深（主板/创业板/科创板）：前收盘价 × (1 ± 涨跌幅限制)，四舍五入到 0.01 元
- 北交所：涨停价向下截断、跌停价向上取整到 0.01 元（保证不超过 ±30%）
涨跌幅限制按非 ST 规则：主板 10%、创业板/科创板 20%、北交所 30%。
历史 ST 日（主板 5%）的收盘价不可能达到 10% 涨停价，因此会被天然排除，不会被误判为涨停。
"""
from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

LIMIT_PCT = {"main": 0.10, "gem": 0.20, "star": 0.20, "bse": 0.30}
EPS = 0.005  # 价格比较容差（半分钱）

# 板型编码：数值越大越强（跌停同理：一字跌停最弱势、倒T字次之）
BT_NONE, BT_SHITI, BT_TZI, BT_YIZI = 0, 1, 2, 3
BT_NAMES = {BT_SHITI: "实体板", BT_TZI: "T字板", BT_YIZI: "一字板"}
BT_DOWN_NAMES = {BT_SHITI: "实体跌停", BT_TZI: "倒T字跌停", BT_YIZI: "一字跌停"}


def round_half_up_cent(x) -> np.ndarray:
    """向量化四舍五入到分。

    prev_close 为两位小数，乘以 1.1/1.2/1.3 后 ×100 的小数部分最多一位，
    +1e-7 足以抵消二进制浮点误差（如 1.265 被存为 1.26499999...），又不会误进位。
    """
    return np.floor(np.asarray(x, dtype="float64") * 100 + 0.5 + 1e-7) / 100


def floor_cent(x) -> np.ndarray:
    return np.floor(np.asarray(x, dtype="float64") * 100 + 1e-7) / 100


def ceil_cent(x) -> np.ndarray:
    return np.ceil(np.asarray(x, dtype="float64") * 100 - 1e-7) / 100


def limit_price_exact(prev_close: float, pct: float, board: str = "main") -> float:
    """十进制精确版（用于测试对照）。pct 为带符号涨跌幅，如 0.1 / -0.1。"""
    v = Decimal(str(prev_close)) * (Decimal(1) + Decimal(str(pct)))
    if board == "bse":
        rounding = ROUND_FLOOR if pct > 0 else ROUND_CEILING
    else:
        rounding = ROUND_HALF_UP
    return float(v.quantize(Decimal("0.01"), rounding=rounding))


def limit_prices(prev_close: pd.Series, board: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    pct = board.map(LIMIT_PCT).astype("float64")
    up_raw = prev_close * (1 + pct)
    dn_raw = prev_close * (1 - pct)
    is_bse = (board == "bse").to_numpy()
    up = np.where(is_bse, floor_cent(up_raw), round_half_up_cent(up_raw))
    dn = np.where(is_bse, ceil_cent(dn_raw), round_half_up_cent(dn_raw))
    return up, dn, pct


def add_limit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """需要列：board, prev_close, open, high, low, close。"""
    up_arr, dn_arr, pct = limit_prices(df["prev_close"], df["board"])
    df["limit_pct"] = pct
    df["up_limit"] = up_arr
    df["down_limit"] = dn_arr
    up, dn = df["up_limit"], df["down_limit"]
    df["is_up"] = (df["close"] - up).abs() < EPS
    df["touch_up"] = (df["high"] - up).abs() < EPS
    df["open_up"] = (df["open"] - up).abs() < EPS
    df["is_down"] = (df["close"] - dn).abs() < EPS
    df["touch_down"] = (df["low"] - dn).abs() < EPS
    df["open_down"] = (df["open"] - dn).abs() < EPS
    yizi = df["is_up"] & ((df["low"] - up).abs() < EPS)
    tzi = df["is_up"] & df["open_up"] & ~yizi
    df["board_type"] = np.select(
        [yizi, tzi, df["is_up"]], [BT_YIZI, BT_TZI, BT_SHITI], default=BT_NONE
    ).astype("int8")
    # 跌停板型：一字（全天在跌停价）、倒T字（开盘跌停、盘中打开、收盘又封回）、实体
    yizi_d = df["is_down"] & ((df["high"] - dn).abs() < EPS)
    tzi_d = df["is_down"] & df["open_down"] & ~yizi_d
    df["board_type_down"] = np.select(
        [yizi_d, tzi_d, df["is_down"]], [BT_YIZI, BT_TZI, BT_SHITI], default=BT_NONE
    ).astype("int8")
    return df
