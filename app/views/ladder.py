"""看板页面：每日涨停天梯。

当日涨停的股票按连板高度分层（最高板在上、首板在下），每层显示只数与晋级率（昨日 h-1 板 → 今日 h 板）；
冲击该层失败的炸板、断板股灰色排在该层后面（可关闭）。同一题材同一种颜色（与明细页一致），鼠标悬停看涨停原因等明细。
页面顶部是当日情绪指标（较前一交易日的变化 + 近 20 个交易日走势）。
数据：scripts/run_backtest.py 生成的 data/features/limitup_ladder.parquet、sentiment.parquet。
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from ashare_bt.analysis import BOARD_NAMES
from ashare_bt.config import FEATURE_DIR, load_config
from ashare_bt.features.ladder import KIND_DK, KIND_ZB, ladder_levels
from ashare_bt.features.limit import BT_NAMES
from ashare_bt.themes import KIND_TOPIC, assign_colors, categories, load_theme_map, main_category, split_tags
from views.ui_common import LADDER_CSS, TREND_DAYS, day_picker, esc, fmt_day, isnan, metric, money, pct, pct_span, pp

LADDER_PATH = FEATURE_DIR / "limitup_ladder.parquet"
SENT_PATH = FEATURE_DIR / "sentiment.parquet"
BT_BADGE = {3: "一字", 2: "T字"}
MARKET_BADGE = {"gem": "创", "star": "科", "bse": "北"}
KIND_TP = "停牌"  # 昨日涨停、今日没有交易（如连板后停牌核查）：用昨日的数据补进天梯
FAIL_ORDER = {KIND_ZB: 0, KIND_DK: 1, KIND_TP: 2}


@st.cache_data(show_spinner="加载涨停天梯…")
def load(m1: float, m2: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(LADDER_PATH), pd.read_parquet(SENT_PATH).set_index("date").sort_index()


st.html(LADDER_CSS)
st.title("每日涨停天梯")
if not (LADDER_PATH.exists() and SENT_PATH.exists()):
    st.error("未找到涨停天梯数据，请先运行 `uv run python scripts/run_backtest.py`。")
    st.stop()
lad, sent = load(LADDER_PATH.stat().st_mtime, SENT_PATH.stat().st_mtime)
universe = load_config()["universe"]
days = [int(d) for d in sent.index if d >= int(lad["date"].min())]

# ---------------------------------------------------------------- 日期：前一/后一交易日按钮 + 日期选择
d = day_picker(days, "ladder_day")
i = days.index(d)
prev = days[i - 1] if i > 0 else None

# ---------------------------------------------------------------- 当日数据与题材
day = lad[lad["date"] == d].copy()
yup = lad[(lad["date"] == prev) & lad["is_up"]] if prev is not None else lad.iloc[0:0]
gone = yup[~yup["code"].isin(set(day["code"]))]  # 昨日涨停、今日没有交易记录 = 停牌
if len(gone):
    nan = float("nan")
    gone = gone.assign(kind=KIND_TP, is_up=False, row=gone["streak"] + 1, prv_streak=gone["streak"], streak=0,
                       prv_reason=gone["reason"], reason=pd.NA, first_time=pd.NA, first_sec=nan,
                       open_pct=nan, close_pct=nan, auc_amt=nan)
    day = pd.concat([day, gone], ignore_index=True)
tm = load_theme_map()
src = day["reason"].fillna(day["prv_reason"])  # 炸板、断板、停牌股当天没有涨停原因：用昨日的归题材
day["tags"] = [split_tags(r) for r in src]
day["cats"] = [categories(t, tm) for t in day["tags"]]
day["main"] = [main_category(c, tm) for c in day["cats"]]
up = day[day["is_up"]]
counts = Counter(m for m in up["main"] if isinstance(m, str))  # 没有涨停原因的为 NaN（pandas 3 里为真值），不计
colors = assign_colors(counts, tm)

# ---------------------------------------------------------------- 情绪指标（较前一交易日 + 近 20 日走势）
win = days[max(0, i - TREND_DAYS + 1): i + 1]
w = lad[lad["date"].isin(win)]
lianban = w[w["is_up"] & (w["streak"] >= 2)].groupby("date").size().reindex(win, fill_value=0)
yzt = w[w["prv_up"]].groupby("date")["close_pct"].mean().reindex(win)
sw = sent.reindex(win)
top = up[up["streak"] == up["streak"].max()] if len(up) else up
cols = st.columns(7)
metric(cols[0], "涨停", sw["zt_count"], lambda v: f"{int(v)}", help="当日收盘封住涨停的家数（股票池内）")
metric(cols[1], "连板", lianban.astype(float), lambda v: f"{int(v)}", help="连板数 ≥ 2 的涨停家数")
metric(cols[2], "最高板", sw["max_height"], lambda v: f"{int(v)} 板",
       help="当日最高连板：" + ("、".join(top["name"].astype(str)) if len(top) else "无"))
metric(cols[3], "炸板率", sw["zb_rate"], lambda v: f"{v:.1%}", diff=pp,
       help="炸板 ÷ 触及涨停（炸板 = 盘中触及涨停价、收盘没封住）")
metric(cols[4], "跌停", sw["dt_count"], lambda v: f"{int(v)}", help="收盘价等于跌停价的家数；明细见左侧「跌停天梯」")
metric(cols[5], "首板晋级率", sw["promo_rate"], lambda v: f"{v:.1%}", diff=pp, help="昨日首板中今日涨停（成为二板）的比例")
metric(cols[6], "昨日涨停溢价", yzt, lambda v: pct(v), diff=pp, help="昨日涨停股今日收盘涨幅的均值（不含停牌）")

# ---------------------------------------------------------------- 控件：层内排序、炸板/断板、题材筛选
k1, k2, _ = st.columns([1.55, 2.4, 4], vertical_alignment="bottom")
SORTS = {"time": "按涨停时间", "theme": "按题材"}
sort = k1.segmented_control("层内排序", list(SORTS), format_func=SORTS.get, default="time", required=True,
                            key="ladder_sort", help="按涨停时间：首次涨停越早越靠前；按题材：同题材排在一起（题材按当日涨停家数排序）")
FAIL_MODES = {"none": "不显示", "high": "3板及以上", "all": "全部"}
fail_mode = k2.segmented_control("未晋级（炸板、断板、停牌）", list(FAIL_MODES), format_func=FAIL_MODES.get,
                                 default="high", required=True, key="ladder_fail_mode",
                                 help="炸板 = 触及涨停价但收盘没封住；断板 = 昨日涨停、今日没有触及涨停价；"
                                      "停牌 = 昨日涨停、今日没有交易。放在所冲击的那一层后面，灰色显示。"
                                      "低位（首板、2板）失败的很多，默认只显示 3 板及以上；各层的失败只数始终显示在层名下")


def show_fail(h: int) -> bool:
    return fail_mode == "all" or (fail_mode == "high" and h >= 3)


hot = [c for c in sorted(counts, key=lambda c: (-counts[c], c)) if counts[c] >= 2 and tm.kind_of(c) == KIND_TOPIC]
sel = (st.pills("题材筛选（数字为当日涨停家数，可多选）", hot, format_func=lambda c: f"{c} {counts[c]}",
                selection_mode="multi", key=f"ladder_themes_{d}") if hot else None) or []
any_mode = st.checkbox("包含次要题材", key="ladder_any", help="其他涨停原因标签属于所选题材的也显示") if sel else False


def matches(r) -> bool:
    return not sel or r.main in sel or (any_mode and any(c in sel for c in r.cats))


theme_order = {c: k for k, c in enumerate(sorted(counts, key=lambda c: (-counts[c], c)))}


def sort_key(r):
    t = r.first_sec if not isnan(r.first_sec) else 1e9
    base = (t, -r.bt, r.code)
    return ((theme_order.get(r.main, 10**6),) + base) if sort == "theme" else base


def fail_key(r):
    """炸板（按首次触板时间）→ 断板（按今日涨幅从高到低）→ 停牌。"""
    t = r.first_sec if r.kind == KIND_ZB and not isnan(r.first_sec) else 1e9
    return (FAIL_ORDER.get(r.kind, 9), t, 0.0 if isnan(r.close_pct) else -r.close_pct, r.code)


def tooltip(r) -> str:
    lines = [f"{r.name} {r.code[4:]}（{BOARD_NAMES.get(r.board, r.board)}）"]
    if r.is_up:
        lines.append(f"{'首板' if r.streak == 1 else f'{r.streak} 连板'} · {BT_NAMES.get(r.bt, '')} · "
                     f"首次涨停 {r.first_time if not isnan(r.first_time) else '-'}")
        lines.append(f"涨停原因：{r.reason if not isnan(r.reason) else '-'}")
    elif r.kind == KIND_TP:
        lines.append(f"停牌：昨日 {r.prv_streak} 板，今日没有交易")
    else:
        h = r.row
        what = "首板" if h == 1 else f"{h} 板"
        lines.append(f"{r.kind}：冲击{what}失败" + (f"（昨日 {r.prv_streak} 板）" if r.prv_streak else "")
                     + (f" · 首次触板 {r.first_time}" if r.kind == KIND_ZB and not isnan(r.first_time) else ""))
    if not r.is_up and not isnan(r.prv_reason):
        lines.append(f"昨日涨停原因：{r.prv_reason}")
    if r.kind != KIND_TP:
        lines.append(f"竞价 {pct(r.open_pct)}" + (f"（9:25 成交 {money(r.auc_amt)}）" if not isnan(r.auc_amt) else "")
                     + f" · 收盘 {pct(r.close_pct)}")
    lines.append(f"{'昨日' if r.kind == KIND_TP else ''}成交额 {money(r.amount)} · "
                 f"换手 {'-' if isnan(r.turnover) else f'{r.turnover:.1%}'} · "
                 f"流通市值 {'-' if isnan(r.float_mv) else f'{r.float_mv:.1f}亿'}")
    return "\n".join(lines)


def tile(r) -> str:
    color = colors.get(r.main)
    badges = ""
    if r.board in MARKET_BADGE:
        badges += f'<i class="mk">{MARKET_BADGE[r.board]}</i>'
    if r.is_up and r.bt in BT_BADGE:
        badges += f'<i class="bt">{BT_BADGE[r.bt]}</i>'
    if not r.is_up:
        badges += f'<i class="{"zb" if r.kind == KIND_ZB else "dk"}">{r.kind}</i>'
    if r.is_up:
        left = esc(r.first_time) if not isnan(r.first_time) else "-"
    elif r.kind == KIND_ZB:
        left = (esc(r.first_time) + " · " if not isnan(r.first_time) else "") + pct_span(r.close_pct)
    elif r.kind == KIND_TP:
        left = "今日停牌"
    else:
        left = pct_span(r.close_pct)
    if isinstance(r.main, str):
        style = f' style="background:{color[0]};color:{color[1]}"' if color else ""
        chip = f'<span class="chip{"" if color else " plain"}"{style}>{esc(r.main)}</span>'
    else:
        chip = ""
    border = f' style="border-left-color:{color[1]}"' if color else ""
    return (f'<div class="t{"" if r.is_up else " fail"}"{border} title="{esc(tooltip(r))}">'
            f'<div class="l1"><b>{esc(r.name)}</b>{badges}<span class="code">{r.code[4:]}</span></div>'
            f'<div class="l2"><span class="tm">{left}</span>{chip}</div></div>')


# ---------------------------------------------------------------- 天梯
levels = ladder_levels(lad, d, prev)
rows_html, shown, top_done = [], 0, False
for lv in levels.itertuples():
    h = lv.height
    sealed = sorted((r for r in up[up["streak"] == h].itertuples() if matches(r)), key=sort_key)
    fails = (sorted((r for r in day[(~day["is_up"]) & (day["row"] == h)].itertuples() if matches(r)), key=fail_key)
             if show_fail(h) else [])
    if sel and not sealed and not fails:
        continue
    shown += len(sealed)
    stats = [f"{lv.n} 只"]
    if not pd.isna(lv.promo_den):
        rate = f" · {lv.promo_num / lv.promo_den:.0%}" if lv.promo_den else ""
        stats.append(f"晋级 {lv.promo_num}/{lv.promo_den}{rate}")
    n_fail = day.loc[~day["is_up"] & (day["row"] == h), "kind"].value_counts()
    if len(n_fail):
        stats.append(" · ".join(f"{k} {int(n_fail[k])}" for k in FAIL_ORDER if k in n_fail))
    tiles = "".join(tile(r) for r in sealed) + "".join(tile(r) for r in fails)
    if not tiles:
        why = f"昨日 {h - 1} 板 {lv.promo_den} 只全部未晋级" if not pd.isna(lv.promo_den) and lv.promo_den else "无"
        hint = "（在「未晋级」里选择显示即可查看）" if not show_fail(h) and len(n_fail) else ""
        tiles = f'<div class="empty">{why}{hint}</div>'
    cls = " top" if lv.n and not top_done else ""
    top_done = top_done or bool(lv.n)
    rows_html.append(f'<div class="r{cls}"><div class="hd"><div class="h">{"首板" if h == 1 else f"{h}板"}</div>'
                     f'<div class="s">{"<br>".join(stats)}</div></div><div class="tiles">{tiles}</div></div>')

n_zb, n_dk = int((day["kind"] == KIND_ZB).sum()), int((day["kind"] == KIND_DK).sum())
st.caption(f"{fmt_day(d)}：涨停 {len(up)} 只（连板 {int((up['streak'] >= 2).sum())} 只，最高 {int(up['streak'].max()) if len(up) else 0} 板），"
           f"炸板 {n_zb} 只，断板 {n_dk} 只（昨日涨停、今日未触板）"
           + (f"，昨日涨停今日停牌 {len(gone)} 只" if len(gone) else "")
           + (f"；题材筛选：{'、'.join(sel)}（{'含次要题材' if any_mode else '按主题材'}），显示涨停 {shown} 只" if sel else ""))
if not rows_html:
    st.info("没有符合筛选条件的股票。")
else:
    st.html(f'<div class="lad">{"".join(rows_html)}</div>')

st.caption(
    "说明：天梯按当日连板数分层（停牌不打断，与行情软件口径一致），最高板在上；「晋级」= 今日 h 板中昨日为 h-1 板的只数 ÷ "
    "昨日 h-1 板只数。炸板 = 盘中触及涨停价、收盘没封住；断板 = 昨日涨停、今日没有触及涨停价；停牌 = 昨日涨停、今日没有交易；"
    "三者放在所冲击的那一层、灰色显示（默认只显示 3 板及以上），题材取昨日的涨停原因。指标卡颜色按 A 股习惯：上升红、下降绿。"
    "卡片左侧色条与题材标签颜色与明细页一致（归类见 `config/theme_groups.yaml`），"
    "鼠标悬停卡片可看涨停原因、竞价、成交额、换手、流通市值。今日的首板、二板、三板就是下一交易日一进二、二进三、三进四的候选，"
    "可在对应明细页「按首板日 / 二板日 / 三板日」查看。"
    f"股票池与明细页一致：剔除 ST、上市不足 {universe['min_list_days']} 个交易日的次新股；数据经 thsdk（日线 + 问财）获取。"
)
