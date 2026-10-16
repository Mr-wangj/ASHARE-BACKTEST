"""看板页面：每日跌停天梯。

当日跌停的股票按连续跌停天数分层（最高连跌在上、首次跌停在下），每层显示只数与续跌率（昨日 h-1 连跌 → 今日 h 连跌）；
冲击该层失败的翘板（触及跌停未封住）、止跌（昨日跌停、今日未触及跌停价）、停牌股灰色排在该层后面。
跌停股当天没有涨停原因：题材取近 20 个交易日最近一次涨停的涨停原因，用来看哪些题材在退潮、高位股有没有补跌。
有快讯数据的日期，悬停卡片可看该股前一交易日收盘后到当日收盘的相关快讯（按「消息规则」标出利空）。
页面顶部是当日亏钱效应指标（较前一交易日的变化 + 近 20 个交易日走势）。
数据：scripts/run_backtest.py 生成的 data/features/limitdown_ladder.parquet（股票池与涨停天梯一致）。
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from ashare_bt.analysis import BOARD_NAMES
from ashare_bt.config import FEATURE_DIR, load_config
from ashare_bt.data import news
from ashare_bt.features.ladder import KIND_QB, KIND_ZD, ladder_levels
from ashare_bt.features.limit import BT_DOWN_NAMES
from ashare_bt.monitor import classify, load_rules
from ashare_bt.themes import KIND_TOPIC, assign_colors, categories, load_theme_map, main_category, split_tags
from views.ui_common import LADDER_CSS, TREND_DAYS, day_picker, esc, fmt_day, isnan, metric, money, pct, pct_span, pp

DOWN_PATH = FEATURE_DIR / "limitdown_ladder.parquet"
UP_PATH = FEATURE_DIR / "limitup_ladder.parquet"
SENT_PATH = FEATURE_DIR / "sentiment.parquet"
KIND_TP = "停牌"     # 昨日跌停、今日没有交易：用昨日的数据补进天梯
FAIL_ORDER = {KIND_QB: 0, KIND_ZD: 1, KIND_TP: 2}
BT_BADGE = {3: "一字", 2: "倒T"}
MARKET_BADGE = {"gem": "创", "star": "科", "bse": "北"}
HIGH_H = 3           # 近 20 日最高板 ≥ 3：高位补跌
RECENT_DAYS = 20     # 题材、近期高度的回看交易日数
MAX_TILES = 240      # 每层最多显示的卡片数（普跌日首次跌停可能上千只）
COUNT_COLS = ["dt", "lian", "max_h", "qb", "tiandi", "high", "ztdt"]


@st.cache_data(show_spinner="加载跌停天梯…")
def load(m1: float, m2: float, m3: float):
    """跌停天梯、近期涨停记录（归题材用）、每日亏钱效应指标、交易日列表。"""
    down = pd.read_parquet(DOWN_PATH)
    up = pd.read_parquet(UP_PATH, columns=["date", "code", "is_up", "streak", "reason"])
    up = up[up["is_up"]].drop(columns="is_up").reset_index(drop=True)
    sent = pd.read_parquet(SENT_PATH).set_index("date").sort_index()
    by, dn = down["date"], down["is_down"]
    stats = pd.DataFrame({
        "dt": dn.groupby(by).sum(),
        "lian": (dn & (down["down_streak"] >= 2)).groupby(by).sum(),
        "max_h": down[dn].groupby("date")["down_streak"].max(),
        "qb": (down["kind"] == KIND_QB).groupby(by).sum(),
        "tiandi": (dn & down["tiandi"]).groupby(by).sum(),
        "high": (dn & (down["max_streak"] >= HIGH_H)).groupby(by).sum(),
        "ztdt": (dn & down["prv_up"]).groupby(by).sum(),
        "ydt": down[down["prv_down"]].groupby("date")["close_pct"].mean(),   # 昨日跌停股今日表现（不含停牌）
    })
    stats["qb_rate"] = stats["qb"] / (stats["dt"] + stats["qb"]).replace(0, np.nan)
    days = [int(d) for d in sent.index if d >= int(down["date"].min())]
    stats = stats.reindex(days)
    stats[COUNT_COLS] = stats[COUNT_COLS].fillna(0)
    return down, up, stats, days


def label(h: int) -> str:
    return "首次跌停" if h == 1 else f"{h}连跌"


st.html(LADDER_CSS)
st.title("每日跌停天梯")
if not (DOWN_PATH.exists() and UP_PATH.exists() and SENT_PATH.exists()):
    st.error("未找到跌停天梯数据，请先运行 `uv run python scripts/run_backtest.py`。")
    st.stop()
down, up_all, stats, days = load(DOWN_PATH.stat().st_mtime, UP_PATH.stat().st_mtime, SENT_PATH.stat().st_mtime)
universe = load_config()["universe"]
pos = {dd: k for k, dd in enumerate(days)}

# ---------------------------------------------------------------- 日期
d = day_picker(days, "dladder_day")
i = pos[d]
prev = days[i - 1] if i > 0 else None

# ---------------------------------------------------------------- 当日数据：跌停 / 翘板 / 止跌，加上昨日跌停今日停牌的
day = down[down["date"] == d].copy()
ydn = down[(down["date"] == prev) & down["is_down"]] if prev is not None else down.iloc[0:0]
gone = ydn[~ydn["code"].isin(set(day["code"]))]   # 昨日跌停、今日没有交易记录 = 停牌
if len(gone):
    nan = float("nan")
    gone = gone.assign(kind=KIND_TP, is_down=False, row=gone["down_streak"] + 1, prv_down_streak=gone["down_streak"],
                       down_streak=0, prv_down=True, prv_up=False, tiandi=False, ditian=False, ths_warp=False,
                       open_pct=nan, close_pct=nan)
    day = pd.concat([day, gone], ignore_index=True)

# 题材：近 20 个交易日最近一次涨停的涨停原因（跌停股当天没有涨停原因；地天板取当天的）
win_up = days[max(0, i - RECENT_DAYS + 1): i + 1]
ru = up_all[up_all["date"].isin(win_up) & up_all["code"].isin(set(day["code"]))].sort_values("date")
last_up = ru.drop_duplicates("code", keep="last").set_index("code")
last_rs = ru.dropna(subset=["reason"]).drop_duplicates("code", keep="last").set_index("code")
day["last_up"] = day["code"].map(last_up["date"])
day["last_reason"] = day["code"].map(last_rs["reason"])
tm = load_theme_map()
day["tags"] = [split_tags(r) for r in day["last_reason"]]
day["cats"] = [categories(t, tm) for t in day["tags"]]
day["main"] = [main_category(c, tm) for c in day["cats"]]
dn = day[day["is_down"]]
counts = Counter(m for m in dn["main"] if isinstance(m, str))  # 没有近期涨停的为 NaN（pandas 3 里为真值），不计
colors = assign_colors(counts, tm, min_count=1)

# 相关快讯：前一交易日 15:00 到当日 15:00、关联到这些股票的（行情复盘类不算）
rules = load_rules()
news_files = sorted(news.FLASH_DIR.glob("*.parquet")) if news.FLASH_DIR.exists() else []
first_news = int(news_files[0].stem) if news_files else None
rel: dict[str, list[tuple]] = {}
if news_files:
    d_dt = datetime.strptime(str(d), "%Y%m%d")
    start = datetime.strptime(str(prev), "%Y%m%d").replace(hour=15) if prev is not None else d_dt
    want = set(day["code"])
    for r in news.load_flash(start, d_dt.replace(hour=15)).itertuples():
        codes = [c for c in (r.codes if isinstance(r.codes, (list, np.ndarray)) else []) if c in want]
        if not codes or (rules.ignore_titles is not None and rules.ignore_titles.search(r.title or "")):
            continue
        hits = classify(f"{r.title} {r.digest}", rules)
        neg = any(cat.weight < 0 for cat, _ in hits)
        for c in codes:
            rel.setdefault(c, []).append((r.time, r.title, [cat.name for cat, _ in hits], neg))
has_neg = {c for c, ms in rel.items() if any(m[3] for m in ms)}

# ---------------------------------------------------------------- 亏钱效应指标（较前一交易日 + 近 20 日走势）
sw = stats.reindex(days[max(0, i - TREND_DAYS + 1): i + 1])
top = dn[dn["down_streak"] == dn["down_streak"].max()] if len(dn) else dn
hi = dn[dn["max_streak"] >= HIGH_H].sort_values(["max_streak", "dd_high"], ascending=[False, True])
td = dn[dn["tiandi"]]
names = lambda df, k=8: "、".join(df["name"].astype(str).head(k)) if len(df) else "无"  # noqa: E731
cols = st.columns(7)
metric(cols[0], "跌停", sw["dt"], lambda v: f"{int(v)}", help="当日收盘价等于跌停价的家数（股票池内）")
metric(cols[1], "连续跌停", sw["lian"], lambda v: f"{int(v)}", help="连续跌停 ≥ 2 天的家数")
metric(cols[2], "最高连跌", sw["max_h"], lambda v: f"{int(v)} 天", help="当日连续跌停天数最多的：" + names(top))
metric(cols[3], "翘板率", sw["qb_rate"], lambda v: f"{v:.1%}", diff=pp,
       help="翘板 ÷ 触及跌停（翘板 = 盘中触及跌停价、收盘没封住）；越高说明跌停价附近承接越强")
metric(cols[4], "天地板", sw["tiandi"], lambda v: f"{int(v)}", help="盘中触及涨停、收盘跌停：" + names(td))
metric(cols[5], "高位补跌", sw["high"], lambda v: f"{int(v)}",
       help=f"近 {RECENT_DAYS} 个交易日最高板 ≥ {HIGH_H} 的跌停股：" + (
           "、".join(f"{n}（曾{s}板）" for n, s in zip(hi["name"].head(8), hi["max_streak"].head(8))) if len(hi) else "无"))
metric(cols[6], "昨日跌停溢价", sw["ydt"], lambda v: pct(v), diff=pp, help="昨日跌停股今日收盘涨跌幅的均值（不含停牌）")

# ---------------------------------------------------------------- 控件：层内排序、未续跌、题材筛选
k1, k2, _ = st.columns([2.3, 2.5, 3.2], vertical_alignment="bottom")
SORTS = {"bt": "按板型", "height": "按近期最高板", "theme": "按题材"}
sort = k1.segmented_control("层内排序", list(SORTS), format_func=SORTS.get, default="bt", required=True,
                            key="dladder_sort",
                            help="按板型：一字跌停 → 倒T字 → 实体（同板型近期连板越高越靠前）；"
                                 "按近期最高板：近 20 日连板越高越靠前（高位补跌在前）；按题材：同题材排在一起")
FAIL_MODES = {"none": "不显示", "high": "2连跌及以上", "all": "全部"}
fail_mode = k2.segmented_control("未续跌（翘板、止跌、停牌）", list(FAIL_MODES), format_func=FAIL_MODES.get,
                                 default="all", required=True, key="dladder_fail_mode",
                                 help="翘板 = 盘中触及跌停价、收盘没封住（含地天板）；止跌 = 昨日跌停、今日没有触及跌停价；"
                                      "停牌 = 昨日跌停、今日没有交易。放在所冲击的那一层后面，灰色显示；各层只数始终显示在层名下")


def show_fail(h: int) -> bool:
    return fail_mode == "all" or (fail_mode == "high" and h >= 2)


hot = [c for c in sorted(counts, key=lambda c: (-counts[c], c)) if tm.kind_of(c) == KIND_TOPIC][:15]
sel = (st.pills("题材筛选（数字为当日跌停家数；题材取近 20 日最近一次涨停的原因，可多选）", hot,
                format_func=lambda c: f"{c} {counts[c]}", selection_mode="multi", key=f"dladder_themes_{d}")
       if hot else None) or []
any_mode = st.checkbox("包含次要题材", key="dladder_any", help="其他涨停原因标签属于所选题材的也显示") if sel else False


def matches(r) -> bool:
    return not sel or r.main in sel or (any_mode and any(c in sel for c in r.cats))


theme_order = {c: k for k, c in enumerate(sorted(counts, key=lambda c: (-counts[c], c)))}


def sort_key(r):
    dd = 0.0 if isnan(r.dd_high) else r.dd_high
    if sort == "height":
        return (-r.max_streak, -r.bt, dd, r.code)
    base = (-r.bt, -r.max_streak, dd, r.code)
    return ((theme_order.get(r.main, 10**6),) + base) if sort == "theme" else base


def fail_key(r):
    """翘板 → 止跌 → 停牌；同类按今日收盘涨幅从高到低（地天板在前）。"""
    return (FAIL_ORDER.get(r.kind, 9), 0.0 if isnan(r.close_pct) else -r.close_pct, r.code)


def tooltip(r) -> str:
    lines = [f"{r.name} {r.code[4:]}（{BOARD_NAMES.get(r.board, r.board)}）"]
    if r.is_down:
        s = f"{label(r.down_streak)} · {BT_DOWN_NAMES.get(r.bt, '')}"
        s += " · 天地板（盘中触及涨停）" if r.tiandi else ""
        s += " · 昨日涨停" if r.prv_up else ""
        lines.append(s)
        if r.ths_warp:
            lines.append("同花顺记为翘板：收盘在跌停价，但盘中打开过、同花顺判定收盘未封住")
    elif r.kind == KIND_TP:
        lines.append(f"停牌：昨日{label(r.prv_down_streak)}，今日没有交易")
    else:
        lines.append(f"{r.kind}：冲击{label(r.row)}失败" + (f"（昨日{label(r.prv_down_streak)}）" if r.prv_down_streak else "")
                     + (" · 地天板（收盘涨停）" if r.ditian else ""))
    hist = f"近 {RECENT_DAYS} 日最高 {r.max_streak} 板" if r.max_streak else f"近 {RECENT_DAYS} 日没有涨停"
    if not isnan(r.last_up):
        ago = i - pos.get(int(r.last_up), i)
        hist += f"，最近涨停 {fmt_day(r.last_up)[5:]}" + (f"（{ago} 个交易日前）" if ago else "（当日）")
    if not isnan(r.dd_high):
        hist += f"；距近期最高收盘 {pct(r.dd_high, 1)}"
    lines.append(hist)
    if not isnan(r.last_reason):
        lines.append(f"最近涨停原因：{r.last_reason}")
    if r.kind != KIND_TP:
        lines.append(f"开盘 {pct(r.open_pct)} · 收盘 {pct(r.close_pct)}")
    lines.append(f"{'昨日' if r.kind == KIND_TP else ''}成交额 {money(r.amount)} · "
                 f"换手 {'-' if isnan(r.turnover) else f'{r.turnover:.1%}'} · "
                 f"流通市值 {'-' if isnan(r.float_mv) else f'{r.float_mv:.1f}亿'}")
    ms = sorted(rel.get(r.code, []), key=lambda m: (not m[3], -m[0].timestamp()))
    if ms:
        lines.append(f"相关快讯（{len(ms)} 条）：")
        for t, title, cats, neg in ms[:3]:
            tag = "［利空］" if neg else (f"［{cats[0]}］" if cats else "")
            lines.append(f"  {t:%m-%d %H:%M} {title}{tag}")
    return "\n".join(lines)


def tile(r) -> str:
    color = colors.get(r.main)
    badges = f'<i class="mk">{MARKET_BADGE[r.board]}</i>' if r.board in MARKET_BADGE else ""
    if r.is_down and r.bt in BT_BADGE:
        badges += f'<i class="btd">{BT_BADGE[r.bt]}</i>'
    if not r.is_down:
        badges += f'<i class="{"zb" if r.kind == KIND_QB else "dk"}">{r.kind}</i>'
    badges += '<i class="td">天地</i>' if r.tiandi else ""
    badges += '<i class="dt">地天</i>' if r.ditian else ""
    badges += '<i class="neg">利空</i>' if r.code in has_neg else ""
    if r.kind == KIND_TP:
        left = "今日停牌"
    elif r.is_down:
        h = (f'<span style="color:#c2410c;font-weight:600">曾{r.max_streak}板</span> · ' if r.max_streak >= HIGH_H
             else (f"曾{r.max_streak}板 · " if r.max_streak >= 2 else ""))
        left = h + f"距高 {pct(r.dd_high, 0)}"
    else:
        left = pct_span(r.close_pct)
    if isinstance(r.main, str):
        style = f' style="background:{color[0]};color:{color[1]}"' if color else ""
        chip = f'<span class="chip{"" if color else " plain"}"{style}>{esc(r.main)}</span>'
    else:
        chip = ""
    border = f' style="border-left-color:{color[1]}"' if color else ""
    return (f'<div class="t{"" if r.is_down else " fail"}"{border} title="{esc(tooltip(r))}">'
            f'<div class="l1"><b>{esc(r.name)}</b>{badges}<span class="code">{r.code[4:]}</span></div>'
            f'<div class="l2"><span class="tm">{left}</span>{chip}</div></div>')


# ---------------------------------------------------------------- 天梯
levels = ladder_levels(down, d, prev, flag="is_down", streak="down_streak")
rows_html, shown, top_done = [], 0, False
for lv in levels.itertuples():
    h = lv.height
    sealed = sorted((r for r in dn[dn["down_streak"] == h].itertuples() if matches(r)), key=sort_key)
    fails = (sorted((r for r in day[(~day["is_down"]) & (day["row"] == h)].itertuples() if matches(r)), key=fail_key)
             if show_fail(h) else [])
    if sel and not sealed and not fails:
        continue
    shown += len(sealed)
    stats_txt = [f"{lv.n} 只"]
    if not pd.isna(lv.promo_den):
        rate = f" · {lv.promo_num / lv.promo_den:.0%}" if lv.promo_den else ""
        stats_txt.append(f"续跌 {lv.promo_num}/{lv.promo_den}{rate}")
    n_fail = day.loc[~day["is_down"] & (day["row"] == h), "kind"].value_counts()
    if len(n_fail):
        stats_txt.append(" · ".join(f"{k} {int(n_fail[k])}" for k in FAIL_ORDER if k in n_fail))
    cards = sealed + fails
    tiles = "".join(tile(r) for r in cards[:MAX_TILES])
    if len(cards) > MAX_TILES:
        tiles += f'<div class="more">…另有 {len(cards) - MAX_TILES} 只未显示（可用题材筛选缩小范围）</div>'
    if not tiles:
        why = (f"昨日{label(h - 1)} {lv.promo_den} 只今日都没有续跌" if not pd.isna(lv.promo_den) and lv.promo_den
               else "无")
        hint = "（在「未续跌」里选择显示即可查看）" if not show_fail(h) and len(n_fail) else ""
        tiles = f'<div class="empty">{why}{hint}</div>'
    cls = " top" if lv.n and not top_done else ""
    top_done = top_done or bool(lv.n)
    rows_html.append(f'<div class="r{cls}"><div class="hd"><div class="h">{label(h)}</div>'
                     f'<div class="s">{"<br>".join(stats_txt)}</div></div><div class="tiles">{tiles}</div></div>')

n_qb, n_zd = int((day["kind"] == KIND_QB).sum()), int((day["kind"] == KIND_ZD).sum())
if not news_files:
    news_txt = ""
elif first_news is not None and d < first_news:
    news_txt = f"；快讯从 {fmt_day(first_news)} 起才有，该日没有相关快讯"
else:
    news_txt = f"；有相关快讯 {sum(c in rel for c in set(day['code']))} 只（含利空 {len(has_neg)} 只）"
st.caption(f"{fmt_day(d)}：跌停 {len(dn)} 只（连续跌停 {int((dn['down_streak'] >= 2).sum())} 只，最高连跌 "
           f"{int(dn['down_streak'].max()) if len(dn) else 0} 天），翘板 {n_qb} 只，止跌 {n_zd} 只（昨日跌停、今日未触及跌停价）"
           + (f"，昨日跌停今日停牌 {len(gone)} 只" if len(gone) else "") + news_txt
           + (f"；题材筛选：{'、'.join(sel)}（{'含次要题材' if any_mode else '按主题材'}），显示跌停 {shown} 只" if sel else ""))

# 亏钱效应小结：高位补跌、天地板、昨涨今跌、跌停集中的题材（对比当日涨停）
parts = []
if len(hi):
    parts.append(f"高位补跌 {len(hi)} 只（" + "、".join(f"{n} 曾{s}板" for n, s in zip(hi["name"].head(6), hi["max_streak"].head(6)))
                 + ("…" if len(hi) > 6 else "") + "）")
if len(td):
    parts.append(f"天地板 {len(td)} 只（{names(td, 6)}）")
zt = dn[dn["prv_up"]]
if len(zt):
    parts.append(f"昨日涨停今日跌停 {len(zt)} 只（{names(zt, 6)}）")
if counts:
    up_today = up_all.loc[up_all["date"] == d, "reason"]
    up_cnt = Counter(main_category(categories(split_tags(r), tm), tm) for r in up_today)
    parts.append("跌停集中的题材：" + "、".join(f"{t} {n}（当日涨停 {up_cnt.get(t, 0)}）" for t, n in counts.most_common(5)))
if parts:
    st.markdown("**亏钱效应**：" + "；".join(parts))

if not rows_html:
    st.info("没有符合条件的股票（当日没有跌停、翘板或止跌）。" if not sel else "没有符合筛选条件的股票。")
else:
    st.html(f'<div class="lad down">{"".join(rows_html)}</div>')

st.caption(
    "说明：天梯按当日连续跌停天数分层（停牌不打断），最高连跌在上；「续跌」= 今日连跌 h 天的只数 ÷ 昨日连跌 h-1 天的只数。"
    "跌停 = 收盘价等于跌停价（同花顺把其中一部分“盘中打开过、收盘未封住”的记为翘板，悬停可见）；"
    "翘板 = 盘中最低价触及跌停价、收盘没封住（含地天板）；止跌 = 昨日跌停、今日没有触及跌停价；停牌 = 昨日跌停、今日没有交易；"
    "三者放在所冲击的那一层、灰色显示。卡片上「曾N板」= 近 20 日最高连板（≥3 橙色，即高位补跌），「距高」= 收盘价距近 20 日"
    "最高收盘价的回撤；题材取近 20 个交易日最近一次涨停的涨停原因（没有近期涨停的不归题材），颜色只在本页内区分题材。"
    "「利空」= 前一交易日收盘后到当日收盘的快讯里，有关联该股、命中「舆情监控 → 消息规则」利空类的消息。"
    "指标卡颜色按 A 股习惯：上升红、下降绿。"
    f"股票池与涨停天梯一致：剔除 ST（ST 股跌停幅度为 5%，按 10% 规则不会被算成跌停）、上市不足 {universe['min_list_days']} "
    "个交易日的次新股。"
)
