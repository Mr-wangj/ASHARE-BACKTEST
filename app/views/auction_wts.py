"""看板页面：竞价弱转强（昨日涨停 × 今日竞价，养家心法口径）。

数据来自 scripts/live_monitor.py 落盘的 data/live/{日期}/auction.parquet（9:25 竞价表），
按「弱转强·抢筹」口径筛选：昨日涨停股中，昨日 13:00 后才封板的尾盘弱板（弱），
今日竞价高开 ≥ 0.2 个涨停且竞价额 ≥ 昨日全天 5%（强）。历史文件没有新列时按
limitup_detail 补算，因此可以回看已有记录的任意一天；所选日期早于最新交易日时，
另取日线给出当日 / 次日实际表现，多日统计用于验证口径的命中率。

弱水三千，只取一瓢：本页是研究口径，不构成投资建议。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from ashare_bt.config import RAW_DIR
from ashare_bt.features.limit import LIMIT_PCT, limit_prices
from ashare_bt.live.context import trading_days
from ashare_bt.live.feed import LIVE_DIR
from views._auction_common import limit_of, load_day

STATS_DAYS = 12          # 多日验证统计回看的交易日数

@st.cache_data(ttl=600)
def _day_perf(day: int) -> pd.DataFrame:
    """当日日线（index = full_code）：open/high/close/prev_close/amount/limit_pct。"""
    d = pd.read_parquet(RAW_DIR / "daily_all.parquet", filters=[("date", "==", day)],
                        columns=["full_code", "open", "high", "close", "prev_close", "amount"])
    if not len(d):
        return pd.DataFrame()
    d = d.set_index("full_code")
    board = pd.Series({c: limit_of(c) for c in d.index})
    up, _, _ = limit_prices(d["prev_close"], board)
    d = d.assign(up_limit=up, limit_pct=board.map(LIMIT_PCT).to_numpy())
    return d


@st.cache_data(ttl=600)
def _next_open(day: int, codes: list[str]) -> pd.Series:
    """命中股票在 day 之后第一个交易日的开盘价（r1 卖出价）。"""
    if not codes:
        return pd.Series(dtype=float)
    end = int(datetime.strptime(str(day), "%Y%m%d").strftime("%Y%m%d")) + 20
    d = pd.read_parquet(RAW_DIR / "daily_all.parquet",
                        filters=[("full_code", "in", codes), ("date", ">", day), ("date", "<=", end)],
                        columns=["full_code", "date", "open"])
    if not len(d):
        return pd.Series(dtype=float)
    return d.sort_values("date").drop_duplicates("full_code").set_index("full_code")["open"]


def hits(t: pd.DataFrame) -> pd.DataFrame:
    """页面口径：昨日涨停股中的 弱转强 / 规则抢筹 / 竞价涨停（一字候选）。"""
    y = t[t["y_up"]]
    return y[y["wts"] | y["grab"] | y["at_limit"]]


def perf_cols(h: pd.DataFrame, day: int) -> pd.DataFrame:
    """当日 / 次日表现（day 早于最新交易日时才有）：T日最高、T日收盘浮盈、T日涨停、T+1开盘。"""
    out = pd.DataFrame(index=h.index)
    d = _day_perf(day)
    if not len(d):
        return out.assign(**{"T日最高%": np.nan, "T日收盘%": np.nan, "T日涨停": pd.NA, "T+1开盘%": np.nan})
    d = d.reindex(h["code"])
    out["T日最高%"] = np.round((d["high"].to_numpy() / d["prev_close"].to_numpy() - 1) * 100, 2)
    out["T日收盘%"] = np.round((d["close"].to_numpy() / d["prev_close"].to_numpy() - 1) * 100, 2)
    sealed = np.abs(d["close"].to_numpy() - d["up_limit"].to_numpy()) < 0.005
    out["T日涨停"] = pd.Series(np.where(sealed, "✓", ""), index=h.index)
    no = _next_open(day, h["code"].tolist()).reindex(h["code"])
    out["T+1开盘%"] = np.round((no.to_numpy() / d["open"].to_numpy() - 1) * 100, 2)
    return out


def day_stats(day: int, cal: list[int]) -> dict:
    t = load_day(day, cal)
    h = hits(t)
    y = t[t["y_up"]]
    out = {"day": day, "y_n": int(len(y)), "hit": int(len(h)),
           "wts": int(t["wts"].sum()), "auc_lim": int(y["at_limit"].sum())}
    d = _day_perf(day)
    if len(h) and len(d):
        d = d.reindex(h["code"])
        sealed = np.abs(d["close"].to_numpy() - d["up_limit"].to_numpy()) < 0.005
        out["sealed"] = int(np.nansum(sealed))
        out["ret"] = float(np.nanmean(d["close"] / d["open"] - 1))
    else:
        out["sealed"], out["ret"] = None, None
    return out


st.title("竞价弱转强")
st.caption("昨日涨停 × 今日竞价（养家心法口径）：弱 = 昨日 13:00 后才封板的尾盘弱板；强 = 今日竞价高开 ≥ 0.2 个涨停"
           "且竞价额 ≥ 昨日全天 5%。抢筹倍数 = 今日竞价额 ÷ 昨日 9:25 竞价额（问财昨日竞价额覆盖不全时为空）。"
           "命中 = 弱转强 / 竞价抢筹 / 竞价涨停（一字候选）。T日收盘% = 当日收盘 ÷ 当日开盘 − 1（开盘买入当天的浮盈，"
           "未扣费）；T+1开盘% 是 r1 口次日卖出收益。回测结论：短线接力扣费后期望为负，本页用于盯盘与研究，不构成投资建议。")

days = sorted((int(p.name) for p in LIVE_DIR.glob("20*") if (p / "auction.parquet").exists()), reverse=True) \
    if LIVE_DIR.exists() else []
if not days:
    st.info("还没有竞价数据。实时监控须在 9:25 前启动才有竞价表（data/live/{日期}/auction.parquet）。")
    st.stop()
cal = trading_days()
today = int(datetime.now().strftime("%Y%m%d"))
day = st.selectbox("交易日（竞价日 = 买入日 T）", days, format_func=lambda d: f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}",
                   key="wts_day")
t = load_day(day, cal)
h = hits(t)
done = day < cal[-1]        # 已收盘、日线已更新：可看当日表现

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("昨日涨停", len(t[t["y_up"]]))
c2.metric("命中（弱转强/抢筹/一字）", len(h))
c3.metric("其中弱转强", int(t["wts"].sum()))
if done and len(h):
    d = _day_perf(day).reindex(h["code"])
    sealed = np.nansum(np.abs(d["close"].to_numpy() - d["up_limit"].to_numpy()) < 0.005)
    c4.metric("T日封板", f"{int(sealed)}/{len(h)}", f"{sealed / len(h):.0%}")
    c5.metric("T日收盘浮盈均值", f"{np.nanmean(d['close'] / d['open'] - 1) * 100:+.1f}%")
else:
    c4.metric("T日封板", "-", help="收盘、日线更新后显示（当日表现看「盘中监控」页）")
    c5.metric("T日收盘浮盈均值", "-")

if not len(h):
    st.info("这一天没有命中（弱转强 / 抢筹 / 一字）的昨日涨停股。")
    st.stop()

theme = h["theme"].fillna("").astype(str).replace("", "其他")
th_cnt = theme.value_counts()
th_wts = h.assign(theme=theme).groupby("theme")["wts"].sum()
th_lim = h.assign(theme=theme).groupby("theme")["at_limit"].sum()

g = h.assign(theme=theme, _cnt=theme.map(th_cnt))
g = g.sort_values(["_cnt", "theme", "auc_vs_yauc"], ascending=[False, True, False], na_position="last")
first_of_group = ~g.duplicated("theme")             # 组内首行显示题材名，形成分组视觉

base = pd.DataFrame({
    "题材": np.where(first_of_group, g["theme"], ""),
    "名称": g["name"], "板数": g["y_streak"], "昨日首停": g["y_first_time"].fillna(""),
    "弱": g["weak"].map({True: "晚板", False: ""}),
    "竞价涨幅%": (g["pct"] * 100).round(2), "竞价额(万)": (g["amount"] / 1e4).round(0),
    "竞价额/昨日%": (g["ratio"] * 100).round(1), "抢筹倍数": g["auc_vs_yauc"].round(1),
    "封单(万)": (g["seal_amt"].fillna(0) / 1e4).round(0),
    "状态": [("竞价涨停" if a else ("弱转强" if w else "抢筹")) for a, w in zip(g["at_limit"], g["wts"])],
})
if done:
    base = pd.concat([base, perf_cols(g, day)], axis=1)

st.markdown(f"**题材汇总**（命中题材 {len(th_cnt)} 个，按命中数排序）")
tsum = pd.DataFrame({"题材": th_cnt.index, "命中": th_cnt.to_numpy(),
                     "弱转强": th_wts.reindex(th_cnt.index).fillna(0).astype(int).to_numpy(),
                     "竞价涨停": th_lim.reindex(th_cnt.index).fillna(0).astype(int).to_numpy()})
if done:
    d = _day_perf(day).reindex(g["code"])
    sealed = pd.Series(np.abs(d["close"].to_numpy() - d["up_limit"].to_numpy()) < 0.005, index=g.index)
    ret = pd.Series(d["close"].to_numpy() / d["open"].to_numpy() - 1, index=g.index)
    gg = g.assign(sealed=sealed, ret=ret)
    tsum["T日封板"] = gg.groupby("theme")["sealed"].sum().reindex(th_cnt.index).fillna(0).astype(int).to_numpy()
    tsum["T日收盘浮盈均值%"] = (gg.groupby("theme")["ret"].mean().reindex(th_cnt.index) * 100).round(1).to_numpy()
st.dataframe(tsum, hide_index=True, width="stretch", height=min(38 + 35 * len(tsum), 360))

st.markdown(f"**命中明细**（{len(h)} 只，按题材分组：命中数多的题材在前，组内按抢筹倍数排序）")
st.dataframe(base, hide_index=True, width="stretch", height=min(38 + 35 * len(h), 700))

with st.expander(f"多日验证（近 {min(STATS_DAYS, len(days))} 个有竞价记录的交易日）"):
    rows = [day_stats(d, cal) for d in days[:STATS_DAYS]]
    s = pd.DataFrame(rows)
    show = pd.DataFrame({
        "日期": s["day"].map(lambda d: f"{str(d)[4:6]}-{str(d)[6:]}"),
        "昨日涨停": s["y_n"], "命中": s["hit"], "弱转强": s["wts"], "竞价涨停": s["auc_lim"],
        "T日封板": s["sealed"], "T日封板率": (s["sealed"] / s["hit"]).map(lambda v: f"{v:.0%}" if v == v else "-"),
        "T日收盘浮盈均值%": s["ret"].map(lambda v: f"{v * 100:+.1f}" if v is not None and v == v else "-"),
    })
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("弱转强口径命中数量一般每天 0～3 只；封板率与浮盈均值用于对照「弱转强是否比普通涨停接力更强」。"
               "样本还很少（实时竞价数据 2026-10-23 起才有积累），结论要等样本足够后再下。")
