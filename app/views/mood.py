"""看板页面：情绪监测——按时间维度观察市场情绪。

日期筛选（开始 / 结束日期，快捷范围近 1 月 ~ 全部，「前一日 / 后一日」整体平移）× 周期（日 / 周 / 月）→
结束日期当天的情绪仪表（14 个指标卡：当日值、较前一交易日、近 20 日走势、近一年分位）→ 情绪温度（含构成）→
大盘（成交额、涨跌家数）→ 涨跌停（家数、炸板率与晋级率）→ 高度与梯队 → 溢价与亏钱效应 → 情绪日历 → 数据表。
周 / 月为期间内的日均值（高度取最高、指数取期末与区间涨跌）。
数据：scripts/run_backtest.py 生成的 data/features/market_mood.parquet（定义见 ashare_bt/features/mood.py）。
"""
from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from datetime import date, datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ashare_bt.config import DATA_DIR, FEATURE_DIR, load_config
from ashare_bt.features.mood import LEVEL_NAMES, TEMP_COMPONENTS, TEMP_NAMES, temp_level
from views.ui_common import DOWN, UP, fmt_day, isnan, metric, pct, pp

MOOD_PATH = FEATURE_DIR / "market_mood.parquet"
PRESETS = {"1m": ("近1月", 22), "3m": ("近3月", 66), "6m": ("近6月", 126), "1y": ("近1年", 250), "all": ("全部", None)}
FREQS = {"D": "日", "W": "周", "M": "月"}
LEVEL_COLORS = ["#2563eb", "#60a5fa", "#9ca3af", "#f97316", "#dc2626"]
LEVEL_SCALE = alt.Scale(domain=LEVEL_NAMES, range=LEVEL_COLORS)
BADGE = {"冰点": "blue", "偏冷": "violet", "中性": "gray", "偏热": "orange", "高潮": "red"}
ORANGE, PURPLE, INDIGO = "#f59e0b", "#8b5cf6", "#6366f1"
UP_L, DOWN_L = "#f87171", "#4ade80"
WEEKDAY = "一二三四五六日"
H = 250
YEAR = 250   # 分位的回看交易日数

st.html("<style>[data-testid='stMainBlockContainer'] { padding-left: 1.75rem; padding-right: 1.75rem; }</style>")


@st.cache_data(show_spinner="加载情绪数据…")
def load(mtime: float) -> pd.DataFrame:
    m = pd.read_parquet(MOOD_PATH)
    m["temp_ma5"] = m["temp"].rolling(5, min_periods=3).mean()
    m["turnover_ma20"] = m["turnover"].rolling(20, min_periods=10).mean()
    return m


def as_date(d: int) -> date:
    return datetime.strptime(str(int(d)), "%Y%m%d").date()


def wan_yi(v) -> str:
    if isnan(v):
        return "-"
    return f"{v / 1e12:.2f} 万亿" if abs(v) >= 1e12 else f"{v / 1e8:,.0f} 亿"


def short(d: int, multi_year: bool) -> str:
    s = fmt_day(d)
    return s[2:] if multi_year else s[5:]


def aggregate(v: pd.DataFrame, freq: str) -> pd.DataFrame:
    """日 → 周 / 月：日均值；高度取最高（个股取最高那天的）；上证取期末收盘与区间涨跌。"""
    v = v.reset_index(drop=True)
    multi = v["date"].iloc[0] // 10000 != v["date"].iloc[-1] // 10000
    if freq == "D":
        wd = pd.to_datetime(v["date"].astype(str)).dt.weekday
        return v.assign(x=[short(d, multi) for d in v["date"]],
                        period=[f"{fmt_day(d)}（周{WEEKDAY[w]}）" for d, w in zip(v["date"], wd)], n_days=1)
    key = pd.to_datetime(v["date"].astype(str)).dt.to_period("W" if freq == "W" else "M")
    g = v.groupby(key, sort=True)
    num = [c for c in v.select_dtypes("number").columns if c != "date"]
    out = g[num].mean()
    for c in ("max_h", "h2", "max_dh"):
        out[c] = g[c].max()
    out["sh_close"] = g["sh_close"].last()
    for c in ("sh_pct", "sz_pct", "gem_pct"):
        if c in v:
            out[c] = g[c].apply(lambda s: (1 + s.dropna()).prod() - 1)
    out["top_names"] = v.loc[g["max_h"].idxmax(), "top_names"].to_numpy()
    out["dtop_names"] = v.loc[g["max_dh"].idxmax(), "dtop_names"].to_numpy()
    out["temp_level"] = out["temp"].map(temp_level)
    d0, d1, n = g["date"].min(), g["date"].max(), g.size()
    out = out.assign(date=d0.to_numpy(), n_days=n.to_numpy()).reset_index(drop=True)
    out["x"] = ([short(d, multi) for d in d0] if freq == "W" else [fmt_day(d)[:7] for d in d0])
    out["period"] = [f"{fmt_day(a)} 至 {fmt_day(b)}（{k} 个交易日）" for a, b, k in zip(d0, d1, n)]
    return out


# ====================================================================== 数据与控件
st.title("情绪监测")
if not MOOD_PATH.exists():
    st.error("未找到情绪数据，请先运行 `uv run python scripts/run_backtest.py`（每日自动更新会生成）。")
    st.stop()
mood = load(MOOD_PATH.stat().st_mtime)
universe = load_config()["universe"]
days = mood["date"].tolist()

# ---------------------------------------------------------------- 日期筛选
# 开始 / 结束日期随时可改；快捷范围以最新交易日为结束；「前一日 / 后一日」把整个区间平移一个交易日。
# 指标卡、温度构成显示结束日期当天。key 带最新交易日：数据更新后默认回到「近 3 月」。
K_START, K_END, K_PRESET = (f"mood_{k}_{days[-1]}" for k in ("start", "end", "preset"))


def as_int(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def snap_up(d: date | None) -> int:
    """不早于 d 的第一个交易日（d 晚于最后一天时取最后一天）。"""
    if d is None:
        return days[0]
    k = bisect_left(days, as_int(d))
    return days[min(k, len(days) - 1)]


def snap_down(d: date | None) -> int:
    """不晚于 d 的最后一个交易日（d 早于第一天时取第一天）。"""
    if d is None:
        return days[-1]
    return days[max(bisect_right(days, as_int(d)) - 1, 0)]


def preset_window(p: str) -> tuple[int, int]:
    n = PRESETS[p][1]
    return (days[0] if n is None else days[max(0, len(days) - n)]), days[-1]


def window() -> tuple[int, int]:
    """当前选择的区间（已对齐到交易日；开始晚于结束时互换）。"""
    s, e = snap_up(st.session_state[K_START]), snap_down(st.session_state[K_END])
    return (s, e) if s <= e else (snap_up(st.session_state[K_END]), snap_down(st.session_state[K_START]))


def set_window(s: int, e: int) -> None:
    st.session_state[K_START], st.session_state[K_END] = as_date(s), as_date(e)


def on_preset() -> None:
    p = st.session_state[K_PRESET]
    if p is not None:   # 再点一次已选中的快捷范围会取消选中：区间保持不变
        set_window(*preset_window(p))


def on_dates() -> None:
    st.session_state[K_PRESET] = None   # 手动改了日期：快捷范围不再高亮


def shift(k: int) -> None:
    s, e = window()
    i, j = days.index(s), days.index(e)
    if k < 0 and j > 0:
        i, j = max(0, i - 1), j - 1
    elif k > 0 and j < len(days) - 1:
        i, j = i + 1, j + 1
    set_window(days[min(i, j)], days[j])
    st.session_state[K_PRESET] = None


if K_END not in st.session_state:
    set_window(*preset_window("3m"))
    st.session_state[K_PRESET] = "3m"
lo, hi = window()
c1, c2, c3, c4, c5, c6 = st.columns([1.25, 1.25, 0.78, 0.78, 3.0, 1.2], vertical_alignment="bottom")
date_kw = {"min_value": as_date(days[0]), "max_value": as_date(days[-1]), "format": "YYYY-MM-DD", "on_change": on_dates}
c1.date_input("开始日期", key=K_START, **date_kw)
c2.date_input("结束日期", key=K_END, help="下方指标卡、温度构成显示结束日期当天", **date_kw)
c3.button("前一日", icon=":material/chevron_left:", key="mood_prev", on_click=shift, args=(-1,), width="stretch",
          help="整个区间前移一个交易日", disabled=hi == days[0])
c4.button("后一日", icon=":material/chevron_right:", icon_position="right", key="mood_next", on_click=shift, args=(1,),
          width="stretch", help="整个区间后移一个交易日", disabled=hi == days[-1])
c5.segmented_control("快捷范围", list(PRESETS), format_func=lambda k: PRESETS[k][0], key=K_PRESET, on_change=on_preset,
                     help="以最新交易日为结束日期；手动改日期后不再高亮")
freq = c6.segmented_control("周期", list(FREQS), format_func=FREQS.get, default="D", required=True, key="mood_freq",
                            help="周 / 月 = 期间内的日均值；涨停高度、跌停高度取期间最高；上证指数取期末收盘")
v_day = mood[mood["date"].between(lo, hi)]
if v_day.empty:
    st.warning("所选区间没有交易日。")
    st.stop()
note = []
if as_int(st.session_state[K_START]) != lo or as_int(st.session_state[K_END]) != hi:
    note.append("已对齐到交易日")
if as_int(st.session_state[K_START]) > as_int(st.session_state[K_END]):
    note.append("开始晚于结束，已互换")
st.caption(f"区间 {fmt_day(lo)} 至 {fmt_day(hi)}，共 {len(v_day)} 个交易日" + (f"（{'，'.join(note)}）" if note else "")
           + f"；下方指标卡为结束日期 {fmt_day(hi)} 当天。数据范围 {fmt_day(days[0])} 至 {fmt_day(days[-1])}。")
as_of = int(v_day["date"].iloc[-1])
hist = mood[mood["date"] <= as_of]          # 截至所选区间最后一天（指标卡、分位、走势都基于它）
cur = hist.iloc[-1]
prv = hist.iloc[-2] if len(hist) >= 2 else None


def qrank(col: str) -> float | None:
    s = hist[col].tail(YEAR).dropna()
    x = cur[col]
    if isnan(x) or len(s) < 20:
        return None
    return float((s < x).mean() + 0.5 * (s == x).mean())


# ====================================================================== 当日情绪仪表
lv = cur["temp_level"] if isinstance(cur["temp_level"], str) else None
wd = WEEKDAY[as_date(as_of).weekday()]
chg5 = cur["temp"] - hist["temp"].iloc[-6] if len(hist) >= 6 and not isnan(cur["temp"]) else float("nan")
head = (f"**{fmt_day(as_of)}（周{wd}）情绪温度 {cur['temp']:.0f}** :{BADGE.get(lv, 'gray')}-badge[{lv}]" if lv
        else f"**{fmt_day(as_of)}（周{wd}）**")
if prv is not None and not isnan(prv["temp"]) and lv:
    head += f"　较前一交易日 {cur['temp'] - prv['temp']:+.0f}" + (f"，较 5 日前 {chg5:+.0f}" if not isnan(chg5) else "")
body = (f"两市成交 {wan_yi(cur['turnover'])}" + (f"（{'放量' if cur['turnover'] >= prv['turnover'] else '缩量'} "
        f"{abs(cur['turnover'] - prv['turnover']) / 1e8:,.0f} 亿）" if prv is not None else "")
        + f"，上涨 {int(cur['n_rise']):,} / 下跌 {int(cur['n_fall']):,}（中位数 {pct(cur['median_pct'])}）；"
        f"涨停 {int(cur['zt'])}、跌停 {int(cur['dt'])}，炸板率 {cur['zb_rate']:.0%}；"
        f"涨停高度 {int(cur['max_h'])} 板（{cur['top_names'] if isinstance(cur['top_names'], str) else '-'}）"
        + (f"，跌停高度 {int(cur['max_dh'])} 天（{cur['dtop_names']}）" if cur["max_dh"] >= 1 else "")
        + (f"；首板晋级率 {cur['promo_fb']:.0%}" if not isnan(cur["promo_fb"]) else "")
        + f"，昨日涨停溢价 {pct(cur['prem_zt'])}。")
st.markdown(head + "<br>" + body, unsafe_allow_html=True)

# 盘中（当日日线还没更新时，用盘中监控的实时统计）
today = int(datetime.now().strftime("%Y%m%d"))
live_p = DATA_DIR / "live" / str(today) / "status.json"
if today > int(mood["date"].iloc[-1]) and live_p.exists():
    s = json.loads(live_p.read_text(encoding="utf-8"))
    ls = s.get("stats") or {}
    if s.get("mode") == "live" and ls.get("n_up") is not None:
        st.info(f"今日{('盘中' if s.get('phase') != '收盘' else '收盘')}（{str(ls.get('time', ''))[11:16]}，来自「盘中监控」，"
                f"口径剔除 ST、含次新股，收盘更新后以日线为准）：涨停 {ls['n_up']}、炸板 {ls.get('n_zb', '-')}"
                + (f"（炸板率 {ls['zb_rate']:.0%}）" if ls.get("zb_rate") is not None else "")
                + f"、跌停 {ls.get('n_down', '-')}；上涨 {ls.get('n_red', '-')} / 下跌 {ls.get('n_green', '-')}"
                + (f"（中位数 {pct(ls['median_pct'])}）" if ls.get("median_pct") is not None else "")
                + (f"；昨日涨停溢价 {pct(ls['yzt_avg'])}" if ls.get("yzt_avg") is not None else ""))

spark = hist.tail(20).set_index("date")


def card(col, label, key, fmt, *, diff=None, help=""):
    q = qrank(key)
    note = f"近一年分位 {q:.0%}" if q is not None else ""
    metric(col, label, spark[key].astype("float64"), fmt, diff=diff, help=(help + ("；" if help and note else "") + note) or None)


i0 = lambda v: f"{int(v)}"  # noqa: E731
r1 = st.columns(7)
card(r1[0], "情绪温度", "temp", lambda v: f"{v:.0f}", diff=lambda v, p: round(v - p),
     help="9 个指标近一年分位的平均（见下方「情绪温度」）")
card(r1[1], "成交额（万亿）", "turnover", lambda v: f"{v / 1e12:.2f}", diff=lambda v, p: f"{(v - p) / 1e8:+,.0f}亿",
     help="沪深两市 A 股成交额合计；较前一交易日的变化单位为亿元")
card(r1[2], "上涨家数", "n_rise", lambda v: f"{int(v):,}", help=f"下跌 {int(cur['n_fall']):,} 家，全 A 股（含 ST、次新、北交所）")
card(r1[3], "涨停家数", "zt", i0, help="收盘封住涨停")
card(r1[4], "跌停家数", "dt", i0, help="收盘价等于跌停价")
card(r1[5], "炸板率", "zb_rate", lambda v: f"{v:.1%}", diff=pp, help="炸板 ÷ 触及涨停")
card(r1[6], "涨停高度", "max_h", lambda v: f"{int(v)} 板",
     help="最高连板：" + (cur["top_names"] if isinstance(cur["top_names"], str) else "无"))
r2 = st.columns(7)
card(r2[0], "跌停高度", "max_dh", lambda v: f"{int(v)} 天",
     help="最高连续跌停：" + (str(cur["dtop_names"]) if cur["max_dh"] >= 1 else "无"))
card(r2[1], "连板家数", "lb", i0, help="连板数 ≥ 2 的涨停家数")
card(r2[2], "首板晋级率", "promo_fb", lambda v: f"{v:.1%}", diff=pp, help="昨日首板中今日涨停的比例")
card(r2[3], "连板晋级率", "promo_lb", lambda v: f"{v:.1%}", diff=pp, help="昨日连板（≥2 板）中今日继续涨停的比例")
card(r2[4], "昨日涨停溢价", "prem_zt", lambda v: pct(v), diff=pp, help="昨日涨停股今日收盘涨跌幅的平均")
card(r2[5], "大面", "damian", i0, help="昨日涨停、今日跌停")
card(r2[6], "天地板", "tiandi", i0, help="盘中触及涨停、收盘跌停")

# ====================================================================== 图表
v = aggregate(v_day, freq)
n_pts = len(v)
X = alt.X("x:O", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelOverlap="greedy", ticks=False))
T_PERIOD = alt.Tooltip("period:N", title="日期" if freq == "D" else "期间")
avg = "" if freq == "D" else "（日均）"
dot = 34 if n_pts <= 90 else 12


def tip(*cols):
    out = [T_PERIOD]
    for c, title, f in cols:
        out.append(alt.Tooltip(f"{c}:Q", title=title, format=f) if f else alt.Tooltip(f"{c}:N", title=title))
    return out


def clamp_domain(neg: pd.Series, pos: pd.Series) -> tuple[alt.Scale, str]:
    """极端日（如 2025-04-07 跌停 2801 家）会把其他日子压扁：超过 98 分位 × 2 的部分截断。"""
    lim_pos = max(float(pos.quantile(0.98)) * 2, 30.0)
    lim_neg = max(float(neg.quantile(0.98)) * 2, 30.0)
    note = ""
    if pos.max() > lim_pos or neg.max() > lim_neg:
        big = v.loc[(pos > lim_pos) | (neg > lim_neg), "period"].tolist()
        note = "（纵轴已截断：" + "、".join(big[:3]) + ("…" if len(big) > 3 else "") + "，数值见悬停）"
        return alt.Scale(domain=[-min(neg.max(), lim_neg), min(pos.max(), lim_pos)], clamp=True), note
    return alt.Scale(), note


def long(cols: list[tuple[str, str, int]]) -> pd.DataFrame:
    """宽表 → 长表：(列, 系列名, 符号)。"""
    frames = [v.assign(series=name, val=v[c] * sign, order=k) for k, (c, name, sign) in enumerate(cols)]
    return pd.concat(frames, ignore_index=True)


def line_chart(cols: list[tuple[str, str, str]], y_title: str, fmt: str, zero_rule: bool = True,
               scale: alt.Scale | None = None):
    """多条折线（图例可点击筛选）：(列, 名称, 颜色)。"""
    d = pd.concat([v.assign(series=name, val=v[c]) for c, name, _ in cols], ignore_index=True)
    sel = alt.selection_point(fields=["series"], bind="legend")
    base = alt.Chart(d).encode(x=X)
    ln = base.mark_line(strokeWidth=1.8, point=alt.OverlayMarkDef(size=dot // 2) if n_pts <= 90 else False).encode(
        y=alt.Y("val:Q", title=y_title, axis=alt.Axis(format=fmt), scale=scale or alt.Scale()),
        color=alt.Color("series:N", scale=alt.Scale(domain=[n for _, n, _ in cols], range=[c for *_, c in cols]),
                        legend=alt.Legend(orient="top", title=None)),
        opacity=alt.when(sel).then(alt.value(1.0)).otherwise(alt.value(0.12)),
        tooltip=[T_PERIOD, alt.Tooltip("series:N", title="指标"), alt.Tooltip("val:Q", title="数值", format=fmt)],
    ).add_params(sel)
    layers = [ln]
    if zero_rule:
        layers.insert(0, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9ca3af", strokeDash=[3, 3]).encode(y="y:Q"))
    return alt.layer(*layers).properties(height=H)


# ---------------------------------------------------------------- 情绪温度
st.subheader("情绪温度")
g1, g2 = st.columns([2.6, 1])
with g1:
    bands = pd.DataFrame({"lo": [0, 20, 40, 60, 80], "hi": [20, 40, 60, 80, 100], "band": LEVEL_NAMES})
    yt = alt.Y("temp:Q", title="情绪温度", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]))
    band = alt.Chart(bands).mark_rect(opacity=0.08).encode(   # 背景色带用单独的颜色比例尺，不影响点的图例
        y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 100])), y2="hi:Q",
        color=alt.Color("band:N", scale=alt.Scale(domain=LEVEL_NAMES, range=LEVEL_COLORS), legend=None))
    base = alt.Chart(v).encode(x=X)
    layers = [band, base.mark_line(color="#9ca3af", strokeWidth=1).encode(y=yt)]
    if freq == "D":
        layers.append(base.mark_line(color=INDIGO, strokeWidth=2.2).encode(y=alt.Y("temp_ma5:Q", title="情绪温度")))
    layers.append(base.mark_circle(size=dot * 1.4, opacity=0.95).encode(
        y=yt, color=alt.Color("temp_level:N", scale=LEVEL_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=tip(("temp", "情绪温度", ".0f"), ("temp_level", "档位", None), ("zt", "涨停" + avg, ".0f"),
                    ("dt", "跌停" + avg, ".0f"), ("zb_rate", "炸板率", ".1%"), ("max_h", "涨停高度", ".0f"),
                    ("prem_zt", "昨日涨停溢价", "+.2%"))))
    st.altair_chart(alt.layer(*layers).resolve_scale(color="independent").properties(height=H + 20), width="stretch")
    st.caption("点为当日温度、颜色为档位" + ("，紫线为 5 日均线（温度日间波动大，看趋势用均线）" if freq == "D" else "")
               + "；背景色带：< 20 冰点、20～40 偏冷、40～60 中性、60～80 偏热、≥ 80 高潮。")
with g2:
    comp = pd.DataFrame({"指标": [TEMP_NAMES[k] for k in TEMP_COMPONENTS],
                         "得分": [None if isnan(cur[f"q_{k}"]) else float(cur[f"q_{k}"]) * 100 for k in TEMP_COMPONENTS]})
    raw = {"zt": f"{int(cur['zt'])} 家", "lb": f"{int(cur['lb'])} 家", "max_h": f"{int(cur['max_h'])} 板",
           "seal_rate": f"{cur['seal_rate']:.0%}" if not isnan(cur["seal_rate"]) else "-",
           "promo_fb": f"{cur['promo_fb']:.0%}" if not isnan(cur["promo_fb"]) else "-", "prem_zt": pct(cur["prem_zt"]),
           "rise_ratio": f"{cur['rise_ratio']:.0%}", "dt": f"{int(cur['dt'])} 家", "damian": f"{int(cur['damian'])} 家"}
    comp["当日值"] = [raw[k] for k in TEMP_COMPONENTS]
    comp["档位"] = comp["得分"].map(temp_level)
    bars = alt.Chart(comp).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("指标:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=150)),
        x=alt.X("得分:Q", scale=alt.Scale(domain=[0, 100]), title="近一年分位得分"),
        color=alt.Color("档位:N", scale=LEVEL_SCALE, legend=None),
        tooltip=["指标", alt.Tooltip("得分:Q", format=".0f"), "当日值"])
    text = alt.Chart(comp).mark_text(align="left", dx=4, fontSize=11).encode(y=alt.Y("指标:N", sort=None), x="得分:Q",
                                                                             text="当日值")
    st.altair_chart(alt.layer(bars, text).properties(height=H + 20, title=f"温度构成（{fmt_day(as_of)[5:]}）"),
                    width="stretch")

# ---------------------------------------------------------------- 大盘
st.subheader("大盘")
g1, g2 = st.columns(2)
with g1:
    vv = v.assign(t_yi=v["turnover"] / 1e8, ma_yi=v["turnover_ma20"] / 1e8)
    base = alt.Chart(vv).encode(x=X)
    color = alt.when(alt.datum.sh_pct >= 0).then(alt.value(UP_L)).otherwise(alt.value(DOWN_L))
    vol = base.mark_bar(opacity=0.8).encode(
        y=alt.Y("t_yi:Q", title="两市成交额（亿）" + avg), color=color,
        tooltip=tip(("t_yi", "两市成交额（亿）" + avg, ",.0f"), ("sh_close", "上证指数", ",.2f"), ("sh_pct", "上证涨跌", "+.2%"),
                    ("gem_pct", "创业板指涨跌", "+.2%")))
    left = [vol]
    if freq == "D":
        left.append(base.mark_line(color="#3b82f6", strokeWidth=1.5, strokeDash=[4, 2]).encode(y="ma_yi:Q"))
    idx = base.mark_line(color=INDIGO, strokeWidth=1.8).encode(
        y=alt.Y("sh_close:Q", title="上证指数", scale=alt.Scale(zero=False), axis=alt.Axis(orient="right")))
    st.altair_chart(alt.layer(alt.layer(*left), idx).resolve_scale(y="independent").properties(
        height=H, title="两市成交额与上证指数"), width="stretch")
    st.caption("柱 = 沪深两市成交额（红：上证上涨日，绿：下跌日）" + ("，蓝虚线 = 20 日均额" if freq == "D" else "")
               + "；紫线 = 上证指数（右轴）。")
with g2:
    d = long([("n_rise", "上涨", 1), ("n_fall", "下跌", -1)])
    m_cnt = float(max(v["n_rise"].max(), v["n_fall"].max()))
    m_med = float(v["median_pct"].abs().max()) or 0.01
    bars = alt.Chart(d).mark_bar(opacity=0.85).encode(
        x=X, y=alt.Y("val:Q", title="家数" + avg, scale=alt.Scale(domain=[-m_cnt, m_cnt]),
                     axis=alt.Axis(labelExpr="abs(datum.value)")),
        color=alt.Color("series:N", scale=alt.Scale(domain=["上涨", "下跌"], range=[UP_L, DOWN_L]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=tip(("n_rise", "上涨", ",.0f"), ("n_fall", "下跌", ",.0f"), ("rise_ratio", "上涨占比", ".1%"),
                    ("median_pct", "涨跌中位数", "+.2%"), ("n_up5", "涨超5%", ",.0f"), ("n_dn5", "跌超5%", ",.0f")))
    med = alt.Chart(v).mark_line(color=INDIGO, strokeWidth=1.8).encode(
        x=X, y=alt.Y("median_pct:Q", title="涨跌中位数", scale=alt.Scale(domain=[-m_med, m_med]),
                     axis=alt.Axis(format=".1%", orient="right")))
    st.altair_chart(alt.layer(bars, med).resolve_scale(y="independent").properties(height=H, title="涨跌家数（赚钱效应）"),
                    width="stretch")
    st.caption("全 A 股（含 ST、次新、北交所）上涨 / 下跌家数；紫线 = 个股涨跌幅中位数（右轴，零轴与左轴对齐）。")

# ---------------------------------------------------------------- 涨跌停
st.subheader("涨跌停")
g1, g2 = st.columns(2)
with g1:
    d = long([("zt", "涨停", 1), ("zb", "炸板", 1), ("dt", "跌停", -1)])
    scale, note = clamp_domain(v["dt"], v["zt"] + v["zb"])
    st.altair_chart(alt.Chart(d).mark_bar(opacity=0.9).encode(
        x=X, y=alt.Y("val:Q", title="家数" + avg, stack="zero", scale=scale, axis=alt.Axis(labelExpr="abs(datum.value)")),
        color=alt.Color("series:N", scale=alt.Scale(domain=["涨停", "炸板", "跌停"], range=[UP, ORANGE, DOWN]),
                        legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("order:Q"),
        tooltip=tip(("zt", "涨停", ".0f"), ("zb", "炸板", ".0f"), ("dt", "跌停", ".0f"), ("zb_rate", "炸板率", ".1%"),
                    ("yizi", "一字涨停", ".0f"), ("early_zt", "10点前涨停", ".0f"))).properties(height=H, title="涨停 / 炸板 / 跌停家数"),
        width="stretch")
    st.caption("向上：涨停（红）+ 炸板（橙，触及涨停未封住）；向下：跌停（绿）" + note + "。")
with g2:
    st.altair_chart(line_chart([("zb_rate", "炸板率", ORANGE), ("promo_fb", "首板晋级率", UP),
                                ("promo_lb", "连板晋级率", PURPLE)], "比例" + avg, ".0%", zero_rule=False)
                    .properties(title="炸板率与晋级率"), width="stretch")
    st.caption("炸板率 = 炸板 ÷ 触及涨停；首板晋级率 = 昨日首板今日涨停的比例；连板晋级率 = 昨日 ≥2 板今日继续涨停的比例。"
               "点击图例可只看某一条。")

# ---------------------------------------------------------------- 高度与梯队
st.subheader("高度与梯队")
g1, g2 = st.columns(2)
with g1:
    d = long([("max_h", "涨停高度", 1), ("max_dh", "跌停高度", -1)])
    hmax = float(max(v["max_h"].max(), v["max_dh"].max(), 3))
    hb = alt.Chart(d).mark_bar(opacity=0.9).encode(
        x=X, y=alt.Y("val:Q", title="板数 / 天数", scale=alt.Scale(domain=[-hmax, hmax]),
                     axis=alt.Axis(labelExpr="abs(datum.value)", tickMinStep=1)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["涨停高度", "跌停高度"], range=[UP, DOWN]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=tip(("max_h", "涨停高度（板）", ".0f"), ("top_names", "最高板", None), ("h2", "次高板", ".0f"),
                    ("max_dh", "跌停高度（天）", ".0f"), ("dtop_names", "最高连跌", None)))
    layers = [hb]
    if n_pts <= 70:
        lab = v.assign(lab=v["max_h"].astype(int).astype(str))
        layers.append(alt.Chart(lab).mark_text(dy=-7, fontSize=10, color=UP).encode(x=X, y="max_h:Q", text="lab:N"))
    st.altair_chart(alt.layer(*layers).properties(height=H, title="涨停高度 / 跌停高度"), width="stretch")
    st.caption("向上：当日最高连板数（红）；向下：当日最高连续跌停天数（绿）。悬停可看个股与次高板（最高板与次高板差得多 = 空间断层）。"
               + ("周 / 月取期间最高。" if freq != "D" else ""))
with g2:
    d = long([("b1", "首板", 1), ("b2", "2板", 1), ("b3", "3板及以上", 1)])
    tot = v["b1"] + v["b2"] + v["b3"]
    lim = max(float(tot.quantile(0.98)) * 2, 30.0)
    over = v.loc[tot > lim, "period"].tolist()
    st.altair_chart(alt.Chart(d).mark_bar(opacity=0.9).encode(
        x=X, y=alt.Y("val:Q", title="涨停家数" + avg, stack="zero",
                     scale=alt.Scale(domain=[0, lim], clamp=True) if over else alt.Scale()),
        color=alt.Color("series:N", scale=alt.Scale(domain=["首板", "2板", "3板及以上"], range=["#fca5a5", "#ef4444", "#991b1b"]),
                        legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("order:Q"),
        tooltip=tip(("b1", "首板", ".0f"), ("b2", "2板", ".0f"), ("b3", "3板及以上", ".0f"), ("lb", "连板合计", ".0f"),
                    ("dt_lb", "连续跌停", ".0f"))).properties(height=H, title="涨停梯队结构"), width="stretch")
    st.caption("当日涨停按连板数分层：首板 / 2板 / 3板及以上；连板占比高 = 接力情绪好"
               + (f"（纵轴已截断：{'、'.join(over[:3])}{'…' if len(over) > 3 else ''}，数值见悬停）" if over else "") + "。")

# ---------------------------------------------------------------- 溢价与亏钱效应
st.subheader("溢价与亏钱效应")
g1, g2 = st.columns(2)
with g1:
    st.altair_chart(line_chart([("prem_zt", "昨日涨停", UP), ("prem_zt_open", "昨日涨停（开盘）", "#fb923c"),
                                ("prem_lb", "昨日连板", PURPLE), ("prem_zb", "昨日炸板", "#a8a29e"),
                                ("prem_dt", "昨日跌停", DOWN)], "今日涨跌幅" + avg, "+.1%")
                    .properties(title="昨日涨停 / 连板 / 炸板 / 跌停股的今日表现"), width="stretch")
    st.caption("各类股票今日收盘涨跌幅的平均（「开盘」= 今日开盘涨跌幅，即竞价溢价）。点击图例可只看某一条。")
with g2:
    loss = v[["damian", "tiandi", "high_fall"]].max(axis=1)
    lim = max(float(loss.quantile(0.98)) * 2, 10.0)
    over = v.loc[loss > lim, "period"].tolist()
    st.altair_chart(line_chart([("damian", "大面", "#16a34a"), ("tiandi", "天地板", PURPLE), ("high_fall", "高位补跌", "#c2410c")],
                               "家数" + avg, ".0f", zero_rule=False,
                               scale=alt.Scale(domain=[0, lim], clamp=True) if over else None).properties(title="亏钱效应"),
                    width="stretch")
    st.caption("大面 = 昨日涨停、今日跌停；天地板 = 盘中触及涨停、收盘跌停；高位补跌 = 近 20 个交易日最高 ≥ 3 板、今日跌停。"
               "越多说明打板、接力资金亏钱越多，高位补跌多 = 高标退潮"
               + (f"（纵轴已截断：{'、'.join(over[:3])}{'…' if len(over) > 3 else ''}，数值见悬停）" if over else "") + "。")

# ---------------------------------------------------------------- 情绪日历
if freq == "D":
    st.subheader("情绪日历")
    cal = v_day.copy()
    dts = pd.to_datetime(cal["date"].astype(str))
    cal["wd"] = [f"周{WEEKDAY[w]}" for w in dts.dt.weekday]
    wk_start = dts - pd.to_timedelta(dts.dt.weekday, unit="D")
    multi = cal["date"].iloc[0] // 10000 != cal["date"].iloc[-1] // 10000
    cal["week"] = wk_start.dt.strftime("%y-%m-%d" if multi else "%m-%d")
    cal["period"] = [f"{fmt_day(d)}（{w}）" for d, w in zip(cal["date"], cal["wd"])]
    heat = alt.Chart(cal).mark_rect(cornerRadius=2, stroke="white", strokeWidth=1).encode(
        x=alt.X("week:O", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelOverlap="greedy", ticks=False, orient="top")),
        y=alt.Y("wd:O", sort=[f"周{c}" for c in WEEKDAY[:5]], title=None),
        color=alt.Color("temp:Q", scale=alt.Scale(scheme="redblue", reverse=True, domain=[0, 100], domainMid=50),
                        legend=alt.Legend(title="情绪温度", orient="right")),
        tooltip=[alt.Tooltip("period:N", title="日期"), alt.Tooltip("temp:Q", title="情绪温度", format=".0f"),
                 alt.Tooltip("temp_level:N", title="档位"), alt.Tooltip("zt:Q", title="涨停"), alt.Tooltip("dt:Q", title="跌停"),
                 alt.Tooltip("max_h:Q", title="涨停高度"), alt.Tooltip("zb_rate:Q", title="炸板率", format=".0%")])
    st.altair_chart(heat.properties(height=150), width="stretch")
    st.caption("每格一个交易日（列 = 周，行 = 周一～周五），颜色 = 情绪温度：蓝冷红热；空格为节假日。")

# ---------------------------------------------------------------- 数据表
with st.expander(f"数据表（{len(v)} {'个交易日' if freq == 'D' else ('周' if freq == 'W' else '个月')}，可在右上角下载 CSV）"):
    tbl = pd.DataFrame({
        "日期" if freq == "D" else "期间": v["period"], "情绪温度": v["temp"].round(0), "档位": v["temp_level"],
        "两市成交额(亿)": (v["turnover"] / 1e8).round(0), "上证": v["sh_close"].round(2), "上证涨跌%": (v["sh_pct"] * 100).round(2),
        "上涨": v["n_rise"].round(0), "下跌": v["n_fall"].round(0), "涨跌中位数%": (v["median_pct"] * 100).round(2),
        "涨停": v["zt"].round(1), "跌停": v["dt"].round(1), "炸板": v["zb"].round(1), "炸板率%": (v["zb_rate"] * 100).round(1),
        "涨停高度": v["max_h"], "最高板": v["top_names"], "次高板": v["h2"], "跌停高度": v["max_dh"], "最高连跌": v["dtop_names"],
        "连板": v["lb"].round(1), "首板晋级%": (v["promo_fb"] * 100).round(1), "连板晋级%": (v["promo_lb"] * 100).round(1),
        "昨涨停溢价%": (v["prem_zt"] * 100).round(2), "昨连板溢价%": (v["prem_lb"] * 100).round(2),
        "昨跌停溢价%": (v["prem_dt"] * 100).round(2), "大面": v["damian"].round(1), "天地板": v["tiandi"].round(1),
        "高位补跌": v["high_fall"].round(1), "一字": v["yizi"].round(1), "10点前涨停": v["early_zt"].round(1),
    }).iloc[::-1]
    st.dataframe(tbl, hide_index=True, width="stretch", height=min(38 + 35 * len(tbl), 560),
                 column_config={"最高板": st.column_config.TextColumn(width="medium"),
                                "最高连跌": st.column_config.TextColumn(width="medium")})

# ---------------------------------------------------------------- 说明与检验
with st.expander("情绪温度怎么算、有没有用"):
    st.markdown(
        "**算法**：下面 9 个指标各自取「当日值在近 250 个交易日里的分位」（跌停家数、大面取反：越多越冷），等权平均后 ×100。"
        "< 20 冰点、20～40 偏冷、40～60 中性、60～80 偏热、≥ 80 高潮。分位是相对近一年而言：同样 60 家涨停，"
        "在冷清的年份算偏热，在火热的年份可能只算中性。\n\n"
        + "、".join(TEMP_NAMES.values()) + "。")
    h = hist[hist["temp"].notna()].copy()
    for c in ("promo_fb", "prem_zt", "sh_pct", "temp", "zt"):
        h[f"n_{c}"] = h[c].shift(-1)
    g = h.groupby("temp_level").agg(天数=("temp", "size"), 涨停=("zt", "mean"), 跌停=("dt", "mean"),
                                    次日首板晋级率=("n_promo_fb", "mean"), 次日昨日涨停溢价=("n_prem_zt", "mean"),
                                    次日上证涨跌=("n_sh_pct", "mean"), 次日温度=("n_temp", "mean")).reindex(LEVEL_NAMES)
    ac = h["temp"].autocorr()
    st.markdown(f"**检验**（{fmt_day(int(h['date'].iloc[0]))} 至 {fmt_day(as_of)}，{len(h)} 个交易日）：各档位当日的平均涨跌停家数，"
                "以及**次日**的首板晋级率、昨日涨停溢价（= 当日涨停股次日的表现）、上证涨跌和温度：")
    st.dataframe(pd.DataFrame({
        "档位": g.index, "天数": g["天数"].fillna(0).astype(int), "当日涨停": g["涨停"].round(0), "当日跌停": g["跌停"].round(0),
        "次日首板晋级率%": (g["次日首板晋级率"] * 100).round(1), "次日昨日涨停溢价%": (g["次日昨日涨停溢价"] * 100).round(2),
        "次日上证涨跌%": (g["次日上证涨跌"] * 100).round(2), "次日温度": g["次日温度"].round(0)}), hide_index=True, width="stretch")
    st.markdown(f"温度一阶自相关 {ac:.2f}：日与日之间变化很大，单日温度更像当天的「体温」，判断情绪周期要看 5 日均线和持续性。"
                "历史上冰点与高潮之后，次日的首板晋级率、昨日涨停溢价都比中间档位高（呈 U 形：冰点后修复、高潮后惯性），"
                "但冰点、高潮各只有二三十天，样本少，只作参考，不构成交易信号。")

st.caption(
    "说明：大盘指标（成交额、涨跌家数、涨跌中位数）为当日有成交的全部 A 股（沪深京，含 ST、次新），两市成交额 = 沪深；"
    f"涨跌停、高度、晋级、溢价、亏钱效应与涨停 / 跌停天梯同一股票池（剔除 ST、上市不足 {universe['min_list_days']} 个交易日的次新股），"
    "涨跌停价按非 ST 规则，连板 / 连跌停牌不打断。「昨日 X 溢价」= 昨日为 X 的股票今日收盘涨跌幅的平均（今日停牌的不计）。"
    "10 点前涨停来自问财首次涨停时间（2024-10 起）。周 / 月视图为期间日均，高度取期间最高。指标卡颜色按 A 股习惯：上升红、下降绿。"
    "数据每日收盘后自动更新（scripts/run_backtest.py → data/features/market_mood.parquet）。"
)
