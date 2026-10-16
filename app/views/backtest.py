"""看板页面：策略回测（一进二·弱转强）。

数据：读取 scripts/run_backtest.py 生成的 data/features/events.parquet，侧栏参数调整后实时重算。
"""
from __future__ import annotations

import math
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from ashare_bt.analysis import AUC_LABELS, BOARD_NAMES, auc_band, quarterly_table, tercile
from ashare_bt.backtest.metrics import by_group, compare_vs_rest, pnl_structure, summarize
from ashare_bt.backtest.portfolio import curve_stats, equity_curve
from ashare_bt.config import FEATURE_DIR, RAW_DIR, REPORT_DIR, load_config
from ashare_bt.features.events import RETURN_RULES, add_cross_section_ranks, compute_returns
from ashare_bt.features.limit import BT_NAMES
from ashare_bt.features.strength import STRENGTH_MODES
from ashare_bt.report import PCT_COLS, RET_COLS
from ashare_bt.strategy.weak_to_strong import SIGNAL_VARIANTS, VARIANT_NAMES, VARIANTS, select

EVENTS_PATH = FEATURE_DIR / "events.parquet"
BOARD_CODES = {v: k for k, v in BOARD_NAMES.items()}
MODE_NAMES = STRENGTH_MODES
SLIPPAGES = [0.0, 0.001, 0.002, 0.003, 0.005, 0.01]


# ---------------------------------------------------------------- 数据
@st.cache_data(show_spinner="加载事件宽表…")
def load_events(mode: str, mtime: float) -> pd.DataFrame:
    ev = pd.read_parquet(EVENTS_PATH)
    if mode != load_config()["strategy"]["strength_mode"]:
        ev = add_cross_section_ranks(ev, mode)
    return ev


@st.cache_data(show_spinner=False)
def load_returns(mode: str, mtime: float, slippage: float) -> pd.DataFrame:
    costs = {**load_config()["costs"], "slippage": slippage}
    return compute_returns(load_events(mode, mtime), costs)


@st.cache_data(show_spinner=False)
def load_calendar() -> list[int]:
    idx = pd.read_parquet(RAW_DIR / "index_daily.parquet")
    return sorted(idx.loc[idx["index"] == "sh", "date"].astype("int64").unique().tolist())


# ---------------------------------------------------------------- 展示工具
def _fmt_for(col: str, s: pd.Series) -> str | None:
    c = str(col)
    if c in PCT_COLS or c.endswith(("封板率", "胜率")):
        return "{:.1%}"
    if c in RET_COLS or c.endswith(("均值", "中位")):
        return "{:+.2%}"
    if "百分点" in c:
        return "{:+.1f}"
    if c.endswith("样本数"):
        return "{:.0f}"
    if c == "p值":
        return "{:.4f}"
    if pd.api.types.is_float_dtype(s):
        return "{:.2f}"
    return None


def show(df: pd.DataFrame, height: int | None = None) -> None:
    if df is None or df.empty:
        st.info("当前参数下没有样本。")
        return
    fmt = {c: f for c in df.columns if (f := _fmt_for(c, df[c]))}
    kwargs = {"height": height} if height else {}
    st.dataframe(df.style.format(fmt, na_rep="-"), hide_index=True, width="stretch", **kwargs)


def pct(x: float, digits: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


# ---------------------------------------------------------------- 侧栏
cfg = load_config()
sp, uni, bt = cfg["strategy"], cfg["universe"], cfg["backtest"]
if not EVENTS_PATH.exists():
    st.error("未找到事件宽表，请先运行：`uv run python scripts/run_backtest.py`")
    st.stop()
mtime = EVENTS_PATH.stat().st_mtime

with st.sidebar:
    st.header("回测参数")
    seg = st.radio("区间", ["全样本", "样本内", "样本外"], horizontal=True,
                   help=f"样本内：T 日 ≤ {bt['in_sample_end']}；样本外：之后")
    mode = st.selectbox("强度排序", list(MODE_NAMES), format_func=MODE_NAMES.get,
                        index=list(MODE_NAMES).index(sp["strength_mode"]) if sp["strength_mode"] in MODE_NAMES else 0,
                        help="time：T-1 首次涨停时间越早越强（与「每日一进二明细」页的 T-1日序号一致）")

    st.subheader("弱转强条件")
    p_str_min = st.slider("弱板：强度分位 P_str ≥", 0.2, 0.9, float(sp["p_str_min"]), 0.05,
                          help="同日首板池内排名分位，0 最强、1 最弱")
    auc_lo, auc_hi = st.slider("竞价标准化涨幅区间", -0.3, 1.0,
                               (float(sp["auc_norm_lo"]), float(sp["auc_norm_hi"])), 0.05,
                               help="竞价涨幅 ÷ 涨跌幅限制；主板 0.2 ≈ 高开 2%，创业板 0.2 ≈ 高开 4%")
    p_auc_max = st.slider("口径A：竞价分位 P_auc ≤", 0.05, 0.5, float(sp["p_auc_max"]), 0.05,
                          help="0 为当日竞价涨幅最高")
    strong_max = st.slider("口径B：强板组 P_str ≤", 0.1, 0.5, float(sp["strong_group_max"]), 0.05,
                           help="弱板竞价涨幅需超过该强板组的竞价涨幅中位数")
    delta_min = st.slider("口径C：Δ = P_str − P_auc ≥", 0.1, 0.8, float(sp["delta_min"]), 0.05)

    st.subheader("交易与组合")
    boards_cn = st.multiselect("板块", list(BOARD_CODES), default=list(BOARD_CODES))
    top_n = int(st.number_input("信号口径每日最多买入（只）", 1, 10, int(sp["top_n"])))
    slip_default = float(cfg["costs"]["slippage"])
    slip_opts = sorted(set(SLIPPAGES + [slip_default]))
    slippage = st.select_slider("单边滑点", options=slip_opts, value=slip_default, format_func=lambda x: f"{x:.1%}")
    rule = st.selectbox("卖出规则（组合、季度表）", list(RETURN_RULES), format_func=lambda r: f"{r}：{RETURN_RULES[r]}")
    st.caption("截面排名基于当日全市场首板池；板块、区间筛选只作用于入选信号。"
               "佣金、印花税、过户费按 config/settings.yaml。")

if not boards_cn:
    st.warning("请至少选择一个板块。")
    st.stop()

# ---------------------------------------------------------------- 计算
ev = load_events(mode, mtime)
rets = load_returns(mode, mtime, slippage)
calendar = load_calendar()
params = {"p_str_min": p_str_min, "p_auc_max": p_auc_max, "strong_group_max": strong_max,
          "delta_min": delta_min, "auc_norm_lo": auc_lo, "auc_norm_hi": auc_hi}
d_min, d_max = int(ev["key_t_date"].min()), int(ev["key_t_date"].max())
is_end = int(bt["in_sample_end"])
date_range = {"全样本": (d_min, d_max), "样本内": (d_min, is_end), "样本外": (is_end + 1, d_max)}[seg]
kw = {"boards": [BOARD_CODES[b] for b in boards_cn], "date_range": date_range}
sel_all = {v: select(ev, v, params, uni["min_pool"], **kw) for v in VARIANTS}
sel_top = {v: select(ev, v, params, uni["min_pool"], top_n=top_n, **kw) for v in VARIANTS}
t_dates = [d for d in calendar if date_range[0] <= d <= date_range[1]]

summ = pd.DataFrame([{"代号": v, "口径": VARIANT_NAMES[v], **summarize(ev, rets, sel_all[v])} for v in VARIANTS])

# ---------------------------------------------------------------- 页头
st.title("策略回测：一进二·弱转强")
st.caption(
    f"数据源 thsdk 日线 · 事件宽表 {len(ev):,} 行 · T 日 {d_min} ~ {d_max} · "
    f"宽表生成于 {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M} · 当前区间：{seg}（{date_range[0]} ~ {date_range[1]}）"
)

b1 = summ.set_index("代号").loc["base1"]
sig = summ[summ["代号"].isin(sorted(SIGNAL_VARIANTS))].dropna(subset=["封板率"])
m1, m2, m3, m4 = st.columns(4)
m1.metric("基准1 封板率", pct(b1["封板率"]), help="全部首板次日竞价买入（剔除开盘即涨停）")
if not sig.empty:
    best = sig.sort_values("封板率", ascending=False).iloc[0]
    _, be = pnl_structure(ev, rets, sel_all[best["代号"]])
    lift = (best["封板率"] - b1["封板率"]) * 100 if not math.isnan(b1["封板率"]) else float("nan")
    m2.metric(f"最高封板率 · {best['口径']}", pct(best["封板率"]),
              delta=None if math.isnan(lift) else f"{lift:+.1f} 个百分点（vs 基准1）")
    m3.metric(f"{best['口径']} · 单笔均值（{rule}）", pct(best[f"{rule}均值"], 2, sign=True),
              help="扣除佣金、印花税、过户费和滑点后的单笔净收益均值")
    m4.metric("盈亏平衡封板率（r1）", pct(be), delta=f"实际 {pct(best['封板率'])}", delta_color="off",
              help="封板率达到该值时，T+1 开盘卖的单笔均值才转正")
else:
    m2.metric("信号口径", "无样本")

tab_over, tab_auc, tab_port, tab_layer, tab_rep = st.tabs(
    ["总览", "竞价分层", "组合回测", "分层分析", "静态报告"]
)

# ---------------------------------------------------------------- 总览
with tab_over:
    st.subheader("各口径结果（全部信号，不限每日只数）")
    show(summ[["口径", "样本数", "交易天数", "封板率", "CI下", "CI上", "触板率", "炸板率", "T日收盘浮盈",
               "r1均值", "r1胜率", "r1盈亏比", "r2均值", "r3均值", "r4均值"]])
    st.caption("封板率 = T 日收盘封住涨停（95% Wilson 区间）。卖出规则："
               + "；".join(f"{k} {v}" for k, v in RETURN_RULES.items()))
    left, right = st.columns(2)
    with left:
        st.subheader("增量检验：信号组 vs 同池其余样本")
        rows = []
        for v in ["A", "B", "C"]:
            for pool_v in (["base1", "base3"] if v != "C" else ["base1"]):
                rows.append({"口径": VARIANT_NAMES[v], "对照池": VARIANT_NAMES[pool_v],
                             **compare_vs_rest(ev, sel_all[v], sel_all[pool_v])})
        show(pd.DataFrame(rows))
        st.caption("对照基准3（弱板 + 同样竞价区间）可以区分：提升来自相对反超，还是只来自高开。")
    with right:
        st.subheader("盈亏结构：封板 vs 未封板")
        pv = st.selectbox("口径", VARIANTS, index=VARIANTS.index("A"), format_func=VARIANT_NAMES.get, key="pnl_v")
        tb, be_v = pnl_structure(ev, rets, sel_all[pv])
        show(tb)
        st.markdown(f"盈亏平衡封板率（r1）：**{pct(be_v)}**；实际封板率：**{pct(ev.loc[sel_all[pv], 'y_t_close_up'].mean())}**")

# ---------------------------------------------------------------- 竞价分层
with tab_auc:
    st.subheader("T 日竞价标准化涨幅 vs 当日表现（按 T-1 强弱拆分）")
    st.caption("样本：当前区间与板块内全部可买入首板（剔除开盘即涨停）。弱板 = P_str ≥ 侧栏阈值。")
    band = auc_band(ev)
    base_idx = sel_all["base1"]
    weak = (ev.loc[base_idx, "x_P_str"] >= p_str_min).to_numpy()
    strong = (ev.loc[base_idx, "x_P_str"] < p_str_min).to_numpy()  # 缺少强度的不计入任一组
    gw = by_group(ev, rets, base_idx[weak], band)
    gs = by_group(ev, rets, base_idx[strong], band)
    metric_sel = st.radio("指标", ["封板率", "触板率", "r1均值", "r3均值"], horizontal=True)
    g = pd.concat([gw.assign(类型="弱板"), gs.assign(类型="强板")], ignore_index=True)
    if not g.empty:
        g["分组"] = g["分组"].astype(str)
        is_ret = metric_sel.endswith("均值")
        chart = alt.Chart(g).mark_bar().encode(
            x=alt.X("分组:N", sort=AUC_LABELS, title="竞价标准化涨幅（竞价涨幅 ÷ 涨跌幅限制）"),
            xOffset=alt.XOffset("类型:N", sort=["弱板", "强板"]),
            y=alt.Y(f"{metric_sel}:Q", title=metric_sel, axis=alt.Axis(format=".1%" if is_ret else ".0%")),
            color=alt.Color("类型:N", scale=alt.Scale(domain=["弱板", "强板"], range=["#1f77b4", "#ff7f0e"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["类型", "分组", "样本数",
                     alt.Tooltip("封板率:Q", format=".1%"), alt.Tooltip("触板率:Q", format=".1%"),
                     alt.Tooltip("r1均值:Q", format="+.2%"), alt.Tooltip("r3均值:Q", format="+.2%")],
        ).properties(height=380)
        st.altair_chart(chart, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**弱板**")
        show(gw)
    with c2:
        st.markdown("**强板**")
        show(gs)

# ---------------------------------------------------------------- 组合回测
with tab_port:
    st.subheader(f"组合净值（{RETURN_RULES[rule]}）")
    st.caption(f"信号口径每日按 Δ 取前 {top_n} 只、等权；基准为当日全部样本等权。"
               "资金分两份按交易日奇偶轮换（竞价卖出的资金赶不上同一场竞价买入）。")
    show_vs = st.multiselect("显示口径", VARIANTS, default=["base1", "base3", "A", "B", "C"],
                             format_func=VARIANT_NAMES.get)
    log_scale = st.toggle("对数坐标", value=True)
    curves, stat_rows = [], []
    for v in show_vs:
        c = equity_curve(ev, rets, sel_top[v], rule, t_dates)
        c["口径"] = VARIANT_NAMES[v]
        curves.append(c)
        stat_rows.append({"口径": VARIANT_NAMES[v] + ("" if v.startswith("base") else f"·前{top_n}"), **curve_stats(c)})
    if curves:
        cdf = pd.concat(curves, ignore_index=True)
        cdf["日期"] = pd.to_datetime(cdf["date"].astype(str), format="%Y%m%d")
        scale = alt.Scale(type="log") if log_scale else alt.Scale(zero=False)
        lines = alt.Chart(cdf).mark_line(strokeWidth=1.6).encode(
            x=alt.X("日期:T", title=None, axis=alt.Axis(format="%Y-%m", labelAngle=0)),
            y=alt.Y("nav:Q", title="净值", scale=scale),
            color=alt.Color("口径:N", legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("日期:T", format="%Y-%m-%d"), "口径", alt.Tooltip("nav:Q", format=".3f", title="净值")],
        )
        one = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((lines + one).properties(height=440), width="stretch")
        show(pd.DataFrame(stat_rows))
    st.subheader(f"按季度（信号口径每日前 {top_n} 只，{rule}）")
    show(quarterly_table(ev, rets, {v: sel_top[v] for v in ["A", "B", "C"]}, rule=rule))

# ---------------------------------------------------------------- 分层分析
with tab_layer:
    dims = {
        "T-1 板型": pd.Series(pd.Categorical(ev["x_bt"].map(BT_NAMES), categories=["一字板", "T字板", "实体板"],
                                            ordered=True), index=ev.index),
        "板块": ev["x_board"].map(BOARD_NAMES),
        "T-1 首板晋级率（三分位）": tercile(ev["x_s_promo_rate_t1"]),
        "T 日昨日涨停竞价均值（三分位）": tercile(ev["x_s_yzt_auc_T"]),
        "T-1 涨停家数（三分位）": tercile(ev["x_s_zt_count_t1"]),
        "T-1 跌停家数（三分位）": tercile(ev["x_s_dt_count_t1"]),
        "T-1 炸板率（三分位）": tercile(ev["x_s_zb_rate_t1"]),
        "T-1 最高连板": ev["x_s_max_height_t1"].clip(upper=9).astype("Int64").astype(str),
        "T-1 流通市值（三分位）": tercile(ev["x_t1_float_mv"], ("小", "中", "大")),
        "T-1 换手率（三分位）": tercile(ev["x_t1_turnover"]),
        "T-1 量比（三分位）": tercile(ev["x_t1_vol_ratio"]),
        "月份": (ev["key_t_date"] // 100).astype(str),
    }
    c1, c2 = st.columns([1, 2])
    lv = c1.selectbox("口径", VARIANTS, index=VARIANTS.index("B"), format_func=VARIANT_NAMES.get, key="layer_v")
    dim = c2.selectbox("分层维度", list(dims))
    tb = by_group(ev, rets, sel_all[lv], dims[dim])
    if not tb.empty:
        tbc = tb.copy()
        tbc["分组"] = tbc["分组"].astype(str)
        order = tbc["分组"].tolist()
        base = alt.Chart(tbc).encode(x=alt.X("分组:N", sort=order, title=dim))
        ch1 = base.mark_bar(color="#1f77b4").encode(
            y=alt.Y("封板率:Q", axis=alt.Axis(format=".0%")),
            tooltip=["分组", "样本数", alt.Tooltip("封板率:Q", format=".1%")]).properties(height=280, title="封板率")
        ch2 = base.mark_bar().encode(
            y=alt.Y("r1均值:Q", axis=alt.Axis(format=".1%"), title="r1 均值"),
            color=alt.condition("datum['r1均值'] > 0", alt.value("#2ca02c"), alt.value("#d62728")),
            tooltip=["分组", "样本数", alt.Tooltip("r1均值:Q", format="+.2%")]).properties(height=280, title="r1 均值")
        cc1, cc2 = st.columns(2)
        cc1.altair_chart(ch1, width="stretch")
        cc2.altair_chart(ch2, width="stretch")
    show(tb)
    st.caption("三分位的切分点按全样本事件计算并固定，不随区间与板块筛选变化。")

# ---------------------------------------------------------------- 静态报告
with tab_rep:
    rp = REPORT_DIR / "mvp_report.md"
    if not rp.exists():
        st.info("尚未生成报告，请运行：`uv run python scripts/run_backtest.py`")
    else:
        st.caption(f"由 scripts/run_backtest.py 按 config/settings.yaml 默认参数生成，"
                   f"更新于 {datetime.fromtimestamp(rp.stat().st_mtime):%Y-%m-%d %H:%M}；不随侧栏参数变化。")
        img = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)$")
        buf: list[str] = []
        for line in rp.read_text(encoding="utf-8").splitlines():
            m = img.match(line.strip())
            if not m:
                buf.append(line)
                continue
            if buf:
                st.markdown("\n".join(buf))
                buf = []
            p = REPORT_DIR / m.group(1)
            if p.exists():
                st.image(str(p))
        if buf:
            st.markdown("\n".join(buf))
