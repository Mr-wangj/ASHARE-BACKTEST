"""涨停天梯、跌停天梯两个页面共用的样式、格式化函数与控件（日期切换、指标卡）。

页面脚本里用 `from views.ui_common import ...` 导入（streamlit run app/dashboard.py 时 app/ 在 sys.path 里）。
"""
from __future__ import annotations

import html
import math
import numbers
from bisect import bisect_right
from datetime import date, datetime

import pandas as pd
import streamlit as st

UP, DOWN = "#e5322d", "#18a058"   # 红涨绿跌，与明细页一致
TREND_DAYS = 20                   # 指标卡走势：近 20 个交易日

LADDER_CSS = """<style>
[data-testid='stMainBlockContainer'] { padding-left: 1.75rem; padding-right: 1.75rem; }
.lad { border: 1px solid rgba(128,128,128,.25); border-radius: 10px; overflow: hidden; }
.lad .r { display: flex; gap: 12px; padding: 9px 12px; border-top: 1px solid rgba(128,128,128,.18); }
.lad .r:first-child { border-top: 0; }
.lad .hd { flex: 0 0 96px; }
.lad .hd .h { font-size: 21px; font-weight: 700; line-height: 1.15; }
.lad .r.top .hd .h { color: #e5322d; }
.lad .hd .s { font-size: 12px; opacity: .7; margin-top: 2px; line-height: 1.35; }
.lad .tiles { flex: 1; min-width: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(152px, 1fr));
  gap: 6px; align-content: start; }
.lad .t { border: 1px solid rgba(128,128,128,.25); border-left: 4px solid rgba(128,128,128,.35); border-radius: 6px;
  padding: 3px 7px 4px; min-width: 0; cursor: default; }
.lad .t.fail { opacity: .55; border-style: dashed; border-left-style: solid; }
.lad .l1 { display: flex; align-items: center; gap: 3px; white-space: nowrap; font-size: 14px; line-height: 20px; }
.lad .l1 b { font-weight: 600; overflow: hidden; text-overflow: ellipsis; }
.lad .l1 i { font-style: normal; font-size: 10.5px; padding: 0 3px; border-radius: 4px; line-height: 15px; }
.lad .l1 .bt { background: #fee2e2; color: #b91c1c; }
.lad .l1 .mk { background: rgba(128,128,128,.16); }
.lad .l1 .zb { background: #fef3c7; color: #92400e; }
.lad .l1 .dk { background: rgba(128,128,128,.22); }
.lad .l1 .code { margin-left: auto; padding-left: 4px; font-size: 11.5px; opacity: .55; }
.lad .l2 { display: flex; align-items: center; justify-content: space-between; gap: 4px; white-space: nowrap;
  font-size: 12px; line-height: 19px; }
.lad .l2 .tm { opacity: .72; font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; }
.lad .chip { padding: 0 6px; border-radius: 9px; font-size: 11.5px; line-height: 18px; max-width: 80px;
  overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
.lad .chip.plain { background: rgba(128,128,128,.14); }
.lad .empty { font-size: 12.5px; opacity: .6; padding-top: 5px; }
.lad .more { font-size: 12.5px; opacity: .6; align-self: center; padding: 0 4px; }
/* 跌停天梯 */
.lad.down .r.top .hd .h { color: #18a058; }
.lad.down .tiles { grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)); }
.lad .l1 .btd { background: #dcfce7; color: #15803d; }
.lad .l1 .td { background: #ede9fe; color: #6d28d9; }
.lad .l1 .dt { background: #fee2e2; color: #b91c1c; }
.lad .l1 .neg { background: #e0f2fe; color: #0369a1; }
</style>"""


def as_date(d: int) -> date:
    return datetime.strptime(str(int(d)), "%Y%m%d").date()


def as_int(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def fmt_day(d: int) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or bool(pd.isna(v))


def pct(v, digits: int = 2) -> str:
    return "-" if isnan(v) else f"{v * 100:+.{digits}f}%"


def pct_span(v) -> str:
    color = "" if isnan(v) or abs(v) < 5e-5 else (UP if v > 0 else DOWN)
    return f'<span style="color:{color}">{pct(v)}</span>' if color else pct(v)


def money(v) -> str:
    if isnan(v):
        return "-"
    return f"{v / 1e8:.2f}亿" if v >= 1e8 else f"{v / 1e4:,.0f}万"


def esc(s) -> str:
    return html.escape("" if isnan(s) else str(s), quote=True)


def pp(v, p) -> str:
    """百分点变化。"""
    return f"{(v - p) * 100:+.1f}pp"


def day_picker(days: list[int], key_prefix: str, label: str = "日期") -> int:
    """「前一日 / 日期 / 后一日」。默认最新交易日（数据更新后自动跳到新的最新交易日）；选到非交易日时显示前一交易日。"""
    key = f"{key_prefix}_{days[-1]}"
    if key not in st.session_state:
        st.session_state[key] = as_date(days[-1])

    def snap(v: date | None) -> int:
        """不晚于所选日期的最近交易日。"""
        if v is None:
            return days[-1]
        return days[max(bisect_right(days, as_int(v)) - 1, 0)]

    def shift(k: int) -> None:
        i = days.index(snap(st.session_state[key])) + k
        st.session_state[key] = as_date(days[min(max(i, 0), len(days) - 1)])

    cur = snap(st.session_state[key])
    c_prev, c_date, c_next, _ = st.columns([0.95, 1.55, 0.95, 5], vertical_alignment="bottom")
    c_prev.button("前一日", icon=":material/chevron_left:", key=f"{key_prefix}_prev", on_click=shift, args=(-1,),
                  help="前一交易日", width="stretch", disabled=cur == days[0])
    picked = c_date.date_input(label, key=key, min_value=as_date(days[0]), max_value=as_date(days[-1]),
                               format="YYYY-MM-DD")
    c_next.button("后一日", icon=":material/chevron_right:", icon_position="right", key=f"{key_prefix}_next",
                  on_click=shift, args=(1,), help="后一交易日", width="stretch", disabled=cur == days[-1])
    d = snap(picked)
    if picked is not None and as_int(picked) != d:
        st.caption(f"{fmt_day(as_int(picked))} 不是交易日，显示前一交易日 {fmt_day(d)}。")
    return d


def metric(col, label: str, series: pd.Series, fmt, *, diff=None, help: str | None = None) -> None:
    """当日值 + 较前一交易日的变化 + 近 20 日走势。颜色按 A 股习惯：上升红、下降绿（inverse）。"""
    v = series.iloc[-1]
    p = series.iloc[-2] if len(series) >= 2 else float("nan")
    delta = None if isnan(v) or isnan(p) else (diff(v, p) if diff else v - p)
    if isinstance(delta, numbers.Real):  # 含 numpy 数值：整数变化显示为整数
        delta = int(delta) if float(delta).is_integer() else float(delta)
    col.metric(label, "-" if isnan(v) else fmt(v), delta=delta, delta_color="inverse", help=help, border=True,
               chart_data=series.fillna(0).tolist(), chart_type="line")
