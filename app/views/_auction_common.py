"""竞价页面的共享数据函数（auction_wts / auction_surprise 用）：读当日竞价表并补算弱转强列。

页面脚本不能互相 import（会执行整个页面渲染），共享逻辑放这里。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ashare_bt.config import RAW_DIR
from ashare_bt.features.limit import LIMIT_PCT
from ashare_bt.live.feed import LIVE_DIR

BOARD_OF = {"60": "main", "00": "main", "30": "gem", "68": "star", "8": "bse", "4": "bse", "9": "bse"}


def board_of(code: str) -> str:
    c = code[4:]
    return BOARD_OF.get(c[:2], "main")


def limit_of(code: str) -> float:
    return LIMIT_PCT[board_of(code)]


def prev_day(day: int, cal: list[int]) -> int | None:
    return cal[cal.index(day) - 1] if cal.index(day) > 0 else None


@st.cache_data(ttl=600)
def _limitup() -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "limitup_detail.parquet",
                           columns=["code", "t1_date", "first_time", "auc_amt_t1"])


def load_day(day: int, cal: list[int]) -> pd.DataFrame:
    """读一天的竞价表并补算弱转强列（engine 新文件已带，旧文件补算，口径一致）。"""
    t = pd.read_parquet(LIVE_DIR / str(day) / "auction.parquet")
    if "y_first_time" not in t.columns:
        pd_ = prev_day(day, cal)
        yest = _limitup()
        yest = yest[yest["t1_date"] == pd_].set_index("code") if pd_ is not None else pd.DataFrame()
        code6 = t["code"].str[4:]
        t["y_first_time"] = code6.map(yest["first_time"]).astype("string") if len(yest) else pd.NA
        t["y_auc_amt"] = code6.map(yest["auc_amt_t1"]).astype("float64") if len(yest) else np.nan
        t = t.reset_index(drop=True)
    lim = pd.Series({c: limit_of(c) for c in t["code"]})
    weak = t["y_first_time"].fillna("") >= "13:00"
    yauc = pd.to_numeric(t["y_auc_amt"], errors="coerce")
    t = t.assign(limit_pct=lim.to_numpy(), weak=weak,
                 auc_vs_yauc=(t["amount"] / yauc.where(yauc > 0)).to_numpy(),
                 wts=(weak & (t["pct"] / lim.to_numpy() >= 0.2) & (t["ratio"] >= 0.05)).to_numpy())
    return t
