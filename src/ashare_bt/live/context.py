"""盘前背景（前一交易日收盘后的本地数据，不调用网络）：

- stocks：每只股票（剔除 ST）昨日的收盘、成交额（及 5 日均额）、涨停 / 连板数 / 炸板 / 跌停 / 连跌、近 20 日最高板，
  以及题材（近 60 个交易日最近一次问财涨停原因的主题材，与明细页、天梯同一套归类）；
- concepts：同花顺概念 / 行业板块的成分股（live/feed.build_concepts 缓存），按规则剔除非题材板块，并映射到题材大类；
- candidates：一进二候选（每日明细里 T-1 = 昨日的首板行，含 T-1 涨停时间、板型等特征），9:25 用竞价算弱转强信号。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import FEATURE_DIR, RAW_DIR
from ..features.limit import LIMIT_PCT, add_limit_columns
from ..monitor import Themer
from ..themes import categories, load_theme_map, main_category, split_tags
from .feed import CONCEPTS_PATH

LOOKBACK = 20     # 近期最高板的回看交易日数
THEME_DAYS = 60   # 题材：近 60 个交易日最近一次涨停原因


@dataclass
class Context:
    day: int                   # 背景对应的交易日（前一交易日 T-1）
    stocks: pd.DataFrame       # index = 个股全码
    concepts: pd.DataFrame     # code, name, kind, n, theme, pos（成分在 stocks 中的行号数组）
    candidates: pd.DataFrame   # 一进二候选
    pos: dict = field(default_factory=dict)   # 全码 → 行号

    @property
    def codes(self) -> list[str]:
        return self.stocks.index.tolist()


def trading_days() -> list[int]:
    idx = pd.read_parquet(RAW_DIR / "index_daily.parquet", columns=["index", "date"])
    return sorted(idx.loc[idx["index"] == "sh", "date"].astype("int64").unique().tolist())


def _runs(flag: pd.Series, codes: pd.Series) -> pd.Series:
    """组内连续为真的天数（按记录，停牌不打断）。"""
    x = flag.astype("int64")
    block = ((codes != codes.shift()) | (x == 0)).cumsum()
    return x.groupby(block).cumsum() * x


def stock_table(day: int, cal: list[int]) -> pd.DataFrame:
    i = cal.index(day)
    win = cal[max(0, i - LOOKBACK + 1): i + 1]
    sec = pd.read_parquet(RAW_DIR / "securities.parquet", columns=["full_code", "name", "board", "is_st_now"])
    sec = sec[~sec["is_st_now"].astype(bool)].drop(columns="is_st_now").set_index("full_code")
    d = pd.read_parquet(RAW_DIR / "daily_all.parquet", filters=[("date", ">=", win[0]), ("date", "<=", day)],
                        columns=["full_code", "date", "open", "high", "low", "close", "amount", "prev_close"])
    d = d[d["full_code"].isin(sec.index)].sort_values(["full_code", "date"]).reset_index(drop=True)
    d["board"] = d["full_code"].map(sec["board"])
    d["prev_close"] = d["prev_close"].fillna(d.groupby("full_code")["close"].shift(1))
    d = add_limit_columns(d)
    d["streak"] = _runs(d["is_up"], d["full_code"])
    d["down_streak"] = _runs(d["is_down"], d["full_code"])
    g = d.groupby("full_code")
    last = d[d["date"] == day].set_index("full_code")
    out = sec.copy()
    out["limit_pct"] = out["board"].map(LIMIT_PCT).astype("float64")
    out["y_close"] = last["close"]
    out["y_open"] = last["open"]     # 判断实时快照的开盘价是否已更新为当日竞价价
    out["y_amount"] = last["amount"]
    out["y_amount5"] = g["amount"].apply(lambda s: s.tail(5).mean())
    out["y_up"] = last["is_up"].reindex(out.index).fillna(False).astype(bool)
    out["y_streak"] = last["streak"].reindex(out.index).fillna(0).astype("int64")
    out["y_zb"] = (last["touch_up"] & ~last["is_up"]).reindex(out.index).fillna(False).astype(bool)
    out["y_down"] = last["is_down"].reindex(out.index).fillna(False).astype(bool)
    out["y_down_streak"] = last["down_streak"].reindex(out.index).fillna(0).astype("int64")
    out["max_h"] = g["streak"].max().reindex(out.index).fillna(0).astype("int64")

    # 题材：近 60 个交易日最近一次问财涨停原因
    lu = pd.read_parquet(RAW_DIR / "limitup_detail.parquet", columns=["code", "t1_date", "reason"])
    lu = lu[lu["t1_date"].between(cal[max(0, i - THEME_DAYS + 1)], day) & lu["reason"].notna()]
    lu = lu.sort_values("t1_date").drop_duplicates("code", keep="last")
    code6 = pd.Series(out.index.str[4:], index=out.index)
    rs = lu.set_index("code")
    out["reason"] = code6.map(rs["reason"]).astype("string")
    out["reason_day"] = code6.map(rs["t1_date"])
    tm = load_theme_map()
    themes = {r: main_category(categories(split_tags(r), tm), tm) for r in pd.unique(out["reason"].dropna())}
    out["theme"] = out["reason"].map(themes).astype("object")
    return out


def concept_table(stocks: pd.DataFrame, rules: dict) -> pd.DataFrame:
    if not CONCEPTS_PATH.exists():
        return pd.DataFrame(columns=["code", "name", "kind", "n", "theme", "pos"])
    sr = rules["sector"]
    c = pd.read_parquet(CONCEPTS_PATH)
    bad = c["name"].str.contains("|".join(map(str, sr["exclude_keywords"])), regex=True)
    c = c[~bad].copy()
    pos = {code: k for k, code in enumerate(stocks.index)}
    c["pos"] = [np.array([pos[m] for m in mem if m in pos], dtype="int64") for mem in c["members"]]
    c["n"] = c["pos"].map(len)
    c = c[c["n"].between(sr["min_size"], sr["max_size"])].drop(columns="members")
    th = Themer(load_theme_map())
    c["theme"] = c["name"].map(th.of_block)
    return c.reset_index(drop=True)


def candidate_table(day: int) -> pd.DataFrame:
    p = FEATURE_DIR / "daily_detail.parquet"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    if "key_level" in d.columns:
        d = d[d["key_level"] == 1]
    return d[d["key_t1_date"] == day].reset_index(drop=True)


def build_context(rules: dict, *, asof: int | None = None) -> Context:
    """asof：背景所在交易日（默认本地数据的最新交易日，即实时监控时的“昨日”）。"""
    cal = trading_days()
    day = asof or cal[-1]
    stocks = stock_table(day, cal)
    return Context(day=day, stocks=stocks, concepts=concept_table(stocks, rules), candidates=candidate_table(day),
                   pos={c: k for k, c in enumerate(stocks.index)})
