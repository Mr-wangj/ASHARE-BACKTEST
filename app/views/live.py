"""看板页面：盘中监控（实时预警）。

数据来自 scripts/live_monitor.py 写入的 data/live/{日期}/（实时监控每 5 秒更新；也可以用 --replay 回放某日分时）。
交易时段本页每 5 秒自动刷新。视图：预警流、板块（强弱与龙头）、竞价（9:25）、一进二信号、涨停监控。
"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime

import pandas as pd
import streamlit as st

from ashare_bt.config import PROJECT_ROOT
from ashare_bt.live.feed import LIVE_DIR
from ashare_bt.live.store import RULES_PATH, read_alerts

LEVEL_ORDER = {"高": 0, "中": 1, "低": 2}
VIEWS = {"alerts": "预警流", "sectors": "板块", "auction": "竞价", "strategy": "一进二信号", "limits": "涨停监控"}
CSS = """<style>
[data-testid='stMainBlockContainer'] { padding-left: 1.75rem; padding-right: 1.75rem; }
.lv .a { display: flex; gap: 10px; align-items: baseline; padding: 6px 2px; border-bottom: 1px solid rgba(128,128,128,.16); }
.lv .tm { flex: 0 0 64px; font-size: 12.5px; opacity: .65; font-variant-numeric: tabular-nums; }
.lv .lv { flex: 0 0 auto; font-size: 11.5px; padding: 0 6px; border-radius: 4px; line-height: 18px; }
.lv .lv.h { background: #fee2e2; color: #b91c1c; } .lv .lv.m { background: #fef3c7; color: #92400e; }
.lv .lv.l { background: rgba(128,128,128,.15); }
.lv .k { flex: 0 0 78px; font-size: 12px; opacity: .8; }
.lv .t { font-weight: 600; font-size: 14px; } .lv .d { font-size: 12.5px; opacity: .72; margin-left: 6px; }
</style>"""


def esc(s) -> str:
    return html.escape("" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s))


def fmt_day(d: int) -> str:
    s = str(d)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def read_status(day: int) -> dict:
    p = LIVE_DIR / str(day) / "status.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_tbl(day: int, name: str) -> pd.DataFrame:
    p = LIVE_DIR / str(day) / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def pct(v, d: int = 1) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:+.{d}f}%"


st.html(CSS)
st.title("盘中监控")
days = sorted((int(p.name) for p in LIVE_DIR.glob("20*") if (p / "status.json").exists()), reverse=True) if LIVE_DIR.exists() else []
if not days:
    st.info("还没有盘中监控数据。实时监控在工作日 9:05 自动启动（收盘后自动退出）；"
            "也可以手动运行 `.venv/bin/python scripts/live_monitor.py`，或用 `--replay 日期` 回放某日分时。")
    st.stop()
today = int(datetime.now().strftime("%Y%m%d"))
c1, c2, _ = st.columns([1.5, 4.5, 2], vertical_alignment="bottom")
day = c1.selectbox("日期", days, key="live_day",
                   format_func=lambda d: fmt_day(d) + ("（回放）" if read_status(d).get("mode") == "replay" else ""))
view = c2.segmented_control("视图", list(VIEWS), format_func=VIEWS.get, default="alerts", required=True, key="live_view")
status = read_status(day)
running = status.get("mode") == "live" and day == today and status.get("phase") not in ("收盘", "非交易日")
if view == "alerts":
    f1, f2, f3 = st.columns([2, 4, 2], vertical_alignment="bottom")
    levels = f1.multiselect("级别", ["高", "中", "低"], default=["高", "中"], key="live_levels")
    kinds = f2.multiselect("类型", ["竞价抢筹", "竞价涨停", "核按钮", "竞价超预期", "竞价强势板块", "一进二信号", "板块异动",
                                    "板块跳水", "同花顺异动", "封涨停", "回封", "炸板", "快速拉升", "大面", "情绪"],
                           key="live_kinds", placeholder="全部类型")
    kw = f3.text_input("搜索", key="live_kw", placeholder="股票 / 板块…")


@st.fragment(run_every=5 if running else None)
def body() -> None:
    s = read_status(day)
    stats = s.get("stats") or {}
    upd = s.get("updated_at", "")
    if s.get("mode") == "replay":
        st.caption(f"回放 {fmt_day(day)}：用当日分时逐分钟重放（{s.get('bars', '-')} 个时点），背景为 {fmt_day(s.get('context_day', day))} "
                   "收盘；分时只有每分钟的价格，盘中瞬间触板、5 秒级拉升会比实时监控少。")
    else:
        age = (datetime.now() - datetime.fromisoformat(upd)).total_seconds() if upd else 1e9
        stale = running and s.get("phase") == "连续竞价" and age > 60
        st.caption(f"{'实时监控' if running else '监控记录'} · 阶段：{s.get('phase', '-')} · 更新于 {upd[11:19] if upd else '-'}"
                   + (f"（本轮 {s['latency']} 秒）" if s.get("latency") is not None else "")
                   + f" · 背景 {fmt_day(s.get('context_day', day))} 收盘，{s.get('n_stocks', '-')} 只股票、"
                   f"{s.get('n_concepts', '-')} 个板块、一进二候选 {s.get('n_candidates', '-')} 只"
                   + (f" · {s['note']}" if s.get("note") else ""))
        if stale:
            st.warning(f"监控已有 {age:.0f} 秒没有更新，可能已停止：查看 data/logs/live_monitor_{day}.log。")
    if stats:
        cols = st.columns(7)
        sp = s.get("special") or {}
        cols[0].metric("涨停", stats.get("n_up", "-"), help="当前封住涨停的家数（剔除 ST）")
        cols[1].metric("炸板", stats.get("n_zb", "-"),
                       help="今日触及涨停、当前没封住的家数" + (f"；炸板率 {stats['zb_rate']:.0%}" if stats.get("zb_rate") is not None else ""))
        cols[2].metric("跌停", stats.get("n_down", "-"))
        cols[3].metric("昨日涨停溢价", pct(stats.get("yzt_avg"), 2),
                       help=f"昨日涨停股当前平均涨幅；继续涨停 {stats.get('yzt_up', '-')} 只"
                            + "".join(f"；同花顺{k}指数 {v * 100:+.2f}%" for k, v in sp.items() if v is not None))
        cols[4].metric("红盘 / 绿盘", f"{stats.get('n_red', '-')} / {stats.get('n_green', '-')}",
                       help=f"涨跌幅中位数 {pct(stats.get('median_pct'), 2)}")
        cols[5].metric("竞价涨停", stats.get("auc_limit", "-"),
                       help=f"竞价 ≥ +5% {stats.get('auc_up5', '-')} 只，≤ -5% {stats.get('auc_dn5', '-')} 只")
        cols[6].metric("竞价抢筹", stats.get("auc_grab", "-"), help=f"昨日涨停股竞价均 {pct(stats.get('auc_yzt'))}")

    if view == "alerts":
        al = read_alerts(day)
        if al.empty:
            st.info("还没有预警。")
            return
        v = al[al["level"].isin(levels or ["高", "中", "低"])]
        if kinds:
            v = v[v["kind"].isin(kinds)]
        if kw:
            m = (v["title"].str.contains(kw, regex=False) | v["detail"].fillna("").str.contains(kw, regex=False)
                 | v["sector"].fillna("").str.contains(kw, regex=False))
            v = v[m]
        v = v.sort_values("time", ascending=False)
        cnt = al["kind"].value_counts()
        st.caption(f"显示 {len(v)} / {len(al)} 条（最新在上，最多 400 条）。各类：" + "、".join(f"{k} {n}" for k, n in cnt.items()))
        cls = {"高": "h", "中": "m", "低": "l"}
        rows = [f'<div class="a"><span class="tm">{r.time:%H:%M:%S}</span><span class="lv {cls.get(r.level, "l")}">{esc(r.level)}</span>'
                f'<span class="k">{esc(r.kind)}</span><span><span class="t">{esc(r.title)}</span>'
                f'<span class="d">{esc(r.detail)}</span></span></div>' for r in v.head(400).itertuples()]
        st.html('<div class="lv">' + "".join(rows) + "</div>")
    elif view == "sectors":
        t = read_tbl(day, "sectors")
        if t.empty:
            st.info("连续竞价开始后才有板块数据。")
            return
        idx = pd.to_numeric(t["idx_pct"], errors="coerce") if "idx_pct" in t else pd.Series(float("nan"), index=t.index)
        strength = idx.where(idx.notna(), t["avg_pct"]) * 100   # 实时监控：同花顺板块指数；回放：成分股等权平均
        st.dataframe(pd.DataFrame({
            "板块": t["name"], "题材": t["theme"].fillna(""), "涨幅%": strength.round(2), "成分均涨%": (t["avg_pct"] * 100).round(2),
            "涨停": t["n_up"], "涨超3%": t["n_strong"], "窗口拉升": t["n_mov"], "空间龙头": t["space"], "先锋（最先涨停）": t["pioneer"],
            "中军（成交额最大）": t["zhongjun"], "领涨": t["top"], "最近集体异动": t["last_alert"], "成分": t["n"],
        }), hide_index=True, width="stretch", height=min(38 + 35 * len(t), 900))
        st.caption("按涨停家数、成分股平均涨幅排序；「涨幅%」= 同花顺板块指数（实时监控时）或成分股等权平均；「窗口拉升」= 最近 3 分钟"
                   "拉升 ≥ 0.2 个涨停或新封板的成分股数。空间龙头 = 封板中、板数最高（并列取先封板的），先锋 = 最先涨停，"
                   "中军 = 涨超 3% 中成交额最大，领涨 = 涨幅最大。板块为同花顺概念 / 行业（剔除融资融券、沪深股通等非题材板块）。")
    elif view == "auction":
        t = read_tbl(day, "auction")
        if t.empty:
            st.info("9:25 竞价结束后才有竞价数据（启动晚于 9:30 时不监控竞价）。")
            return
        base = pd.DataFrame({"名称": t["name"], "题材": t["theme"].fillna(""), "竞价涨幅%": (t["pct"] * 100).round(2),
                             "竞价额(万)": (t["amount"] / 1e4).round(0), "竞价额/昨日%": (t["ratio"] * 100).round(1),
                             "昨日板数": t["y_streak"].where(t["y_up"], 0)})
        g1, g2 = st.columns(2)
        with g1:
            st.markdown(f"**竞价抢筹**（{int(t['grab'].sum())} 只，按 竞价额 ÷ 昨日全天成交额 排序）")
            st.dataframe(base[t["grab"]].sort_values("竞价额/昨日%", ascending=False), hide_index=True, width="stretch", height=420)
        with g2:
            st.markdown(f"**昨日涨停股竞价**（{int(t['y_up'].sum())} 只；核按钮 {int(t['nuke'].sum())}、超预期 {int(t['beyond'].sum())}）")
            y = base[t["y_up"]].assign(状态=[("竞价涨停" if a else ("核按钮" if n else ("超预期" if b else "")))
                                            for a, n, b in zip(t["at_limit"][t["y_up"]], t["nuke"][t["y_up"]], t["beyond"][t["y_up"]])])
            st.dataframe(y.sort_values("竞价涨幅%", ascending=False), hide_index=True, width="stretch", height=420)
        lim = t[t["at_limit"]]
        st.markdown(f"**竞价涨停**（{len(lim)} 只，一字候选；封单 = 9:25 撮合后未成交的涨停价买单）")
        st.dataframe(pd.DataFrame({"名称": lim["name"], "题材": lim["theme"].fillna(""), "昨日板数": lim["y_streak"].where(lim["y_up"], 0),
                                   "竞价额(万)": (lim["amount"] / 1e4).round(0), "封单(万)": (lim["seal_amt"] / 1e4).round(0)})
                     .sort_values("封单(万)", ascending=False), hide_index=True, width="stretch")
    elif view == "strategy":
        t = read_tbl(day, "strategy")
        if t.empty:
            st.info("9:25 竞价结束后计算（需要昨日首板候选：每日明细里的首板行）。")
            return
        sig = t[["sig_A", "sig_B", "sig_C"]].any(axis=1)
        st.caption(f"一进二候选 {len(t)} 只（昨日首板），信号 {int(sig.sum())} 只，每日前 2（按 Δ）{int(t['sig_top'].sum())} 只。"
                   "口径与「策略回测」页相同：弱板 = T-1 强度分位 P_str ≥ 0.5（首次涨停越晚越弱），竞价标准化涨幅在 0.2~0.7 之间；"
                   "A = 竞价分位 P_auc ≤ 0.2，B = 竞价超过强板组中位数，C = Δ = P_str - P_auc ≥ 0.4。"
                   "回测里信号组封板率明显高于基准，但扣费后单笔期望为负，只适合作为盯盘线索。")
        st.dataframe(pd.DataFrame({
            "名称": t["key_name"], "T-1序号": t["x_t1_rank"].astype("Int64").astype(str) + "/" + t["x_pool_n"].astype("Int64").astype(str),
            "T-1涨停时间": t["x_t1_first_time"], "涨停原因": t["x_t1_reason"], "竞价涨幅%": (t["x_auc_pct"] * 100).round(2),
            "竞价额(万)": (t["x_t_auc_amt"] / 1e4).round(0), "T日序号": t["x_t_rank"].astype("Int64"),
            "P_str": t["x_P_str"].round(2), "P_auc": t["x_P_auc"].round(2), "Δ": t["x_delta"].round(2),
            "开盘涨停": t["x_open_limit"].map({True: "是", False: ""}), "A": t["sig_A"].map({True: "✓", False: ""}),
            "B": t["sig_B"].map({True: "✓", False: ""}), "C": t["sig_C"].map({True: "✓", False: ""}),
            "前2": t["sig_top"].map({True: "★", False: ""}),
        }), hide_index=True, width="stretch", height=min(38 + 35 * len(t), 760),
            column_config={"涨停原因": st.column_config.TextColumn(width="medium")})
    else:
        t = read_tbl(day, "limits")
        if t.empty:
            st.info("还没有触及涨停的股票。")
            return
        st.caption(f"今日触及涨停 {len(t)} 只：封板 {int(t['sealed'].sum())}、炸板 {int((~t['sealed']).sum())}。"
                   "首次涨停时间：实时监控取同花顺统计（秒），回放为分钟。")
        st.dataframe(pd.DataFrame({
            "名称": t["name"], "板数": t["board"].map(lambda b: "首板" if b == 1 else f"{b}板"), "首次涨停": t["first_up"],
            "状态": t["sealed"].map({True: "封板", False: "炸板"}), "涨幅%": (t["pct"] * 100).round(2),
            "成交额(亿)": (t["amount"] / 1e8).round(2), "题材": t["theme"].fillna(""), "龙头": t["leader_of"],
        }), hide_index=True, width="stretch", height=min(38 + 35 * len(t), 800))


body()
st.caption(f"预警规则：`{RULES_PATH.relative_to(PROJECT_ROOT)}`（阈值、级别、通知、冷却时间，修改后监控每分钟重新读取）。"
           "高级别预警弹 macOS 通知（20 秒内的多条合并）。实时监控工作日 9:05 自动启动、收盘后退出；"
           "非交易日自动退出。日志：data/logs/live_monitor_日期.log。")
