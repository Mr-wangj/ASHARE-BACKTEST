"""看板页面：舆情监控（按题材）——明日题材预判、今日涨停原因、盘后舆情、消息规则。

- 明日预判：题材的量价基础概率（历史回测校准）+ 盘后催化（快讯按消息定义打分、事件日历、问财热门首次上榜、人气榜）。
- 涨停原因：同花顺涨停分析的“行业原因 / 公司原因”，按来源（公告、互动易、研报、政策、媒体、传闻大V）分类，
  区分当日催化（3 天内的消息）与题材属性（更早的背景）。
- 盘后舆情：当日 15:00 到次日 9:25 的快讯，可按消息类别、题材、重要性筛选；热门概念 / 行业、人气榜、事件日历。
- 消息规则：config/news_rules.yaml 的消息定义（哪些消息影响股价、权重多大），修改后刷新生效。
数据：scripts/download_news.py（每日自动更新时运行）。
"""
from __future__ import annotations

import html
import math
from bisect import bisect_right
from collections import Counter
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

from ashare_bt.config import FEATURE_DIR, PROJECT_ROOT
from ashare_bt.data import news
from ashare_bt.monitor import (FEATURES, MODEL_PATH, RULES_PATH, TARGET_N, Themer, build_theme_panel, day_prediction,
                               explode_themes, load_model, load_rules, parse_reason, review_predictions, train_model)
from ashare_bt.themes import GROUPS_PATH, load_theme_map

LADDER_PATH = FEATURE_DIR / "limitup_ladder.parquet"
SENT_PATH = FEATURE_DIR / "sentiment.parquet"
SRC_ORDER = ["政策", "公告", "研报", "互动易", "媒体", "传闻大V", "产业消息", "其他"]
UP, DOWN = "#e5322d", "#18a058"

CSS = """<style>
[data-testid='stMainBlockContainer'] { padding-left: 1.75rem; padding-right: 1.75rem; }
.mon .chip { display: inline-block; padding: 0 7px; border-radius: 9px; font-size: 11.5px; line-height: 18px;
  margin: 1px 4px 1px 0; background: rgba(128,128,128,.14); white-space: nowrap; }
.mon .chip.pos { background: #fee2e2; color: #b91c1c; } .mon .chip.neg { background: #dcfce7; color: #15803d; }
.mon .chip.imp { background: #fef3c7; color: #92400e; }
.mon .msg { padding: 7px 2px; border-bottom: 1px solid rgba(128,128,128,.16); }
.mon .msg .hd { display: flex; gap: 8px; align-items: baseline; }
.mon .msg .tm { font-size: 12px; opacity: .6; font-variant-numeric: tabular-nums; flex: 0 0 74px; }
.mon .msg a { color: inherit; text-decoration: none; font-weight: 600; }
.mon .msg a:hover { text-decoration: underline; }
.mon .msg .dg { font-size: 12.5px; opacity: .72; margin: 2px 0 2px 82px; line-height: 1.45; }
.mon .msg .tg { margin-left: 82px; }
.mon .item { font-size: 13px; line-height: 1.5; padding: 3px 0; }
.mon .item .n { opacity: .6; font-size: 12px; margin-left: 4px; }
</style>"""


def mtime(p) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def as_date(d: int) -> date:
    return datetime.strptime(str(int(d)), "%Y%m%d").date()


def as_int(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def fmt_day(d: int) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def esc(s) -> str:
    return html.escape("" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s), quote=True)


def chips(items, cls: str = "") -> str:
    return "".join(f'<span class="chip {cls}">{esc(x)}</span>' for x in items if x)


@st.cache_data(show_spinner="加载题材数据…")
def load_base(m_lad: float, m_sent: float, m_tm: float):
    lad = pd.read_parquet(LADDER_PATH)
    sent = pd.read_parquet(SENT_PATH).set_index("date").sort_index()
    themes, daily, panel = build_theme_panel(lad, sent, load_theme_map())
    return lad, sent, themes, daily, panel


@st.cache_data(show_spinner="训练基础模型…")
def model_for(m_model: float, m_lad: float, m_tm: float, split: int):
    m = load_model()
    if m is not None:
        return m
    _lad, _sent, _themes, _daily, panel = load_base(m_lad, mtime(SENT_PATH), m_tm)
    return train_model(panel, split)


@st.cache_data(ttl=60, show_spinner="计算明日预判…")
def prediction_for(day: int, next_day: int | None, m_lad: float, m_tm: float, m_rules: float, m_news: float):
    lad, sent, themes, _daily, panel = load_base(m_lad, mtime(SENT_PATH), m_tm)
    model = model_for(mtime(MODEL_PATH), m_lad, m_tm, 20251231)
    return day_prediction(day, next_day, ladder=lad, themes=themes, panel=panel, model=model, rules=load_rules(),
                          tm=load_theme_map())


st.html(CSS)
st.title("舆情监控")
if not (LADDER_PATH.exists() and SENT_PATH.exists()):
    st.error("未找到涨停天梯数据，请先运行 `uv run python scripts/run_backtest.py`。")
    st.stop()
if not news.FLASH_DIR.exists():
    st.error("还没有舆情数据，请先运行 `uv run python scripts/download_news.py --backfill 10`。")
    st.stop()

m_lad, m_tm, m_rules = mtime(LADDER_PATH), mtime(GROUPS_PATH), mtime(RULES_PATH)
lad, sent, themes, daily, panel = load_base(m_lad, mtime(SENT_PATH), m_tm)
model = model_for(mtime(MODEL_PATH), m_lad, m_tm, 20251231)
rules, tm = load_rules(), load_theme_map()
days = sorted(int(d) for d in panel["date"].unique())

# ---------------------------------------------------------------- 日期
key = f"mon_day_{days[-1]}"
if key not in st.session_state:
    st.session_state[key] = as_date(days[-1])


def snap(v) -> int:
    return days[-1] if v is None else days[max(bisect_right(days, as_int(v)) - 1, 0)]


def shift(k: int) -> None:
    i = days.index(snap(st.session_state[key])) + k
    st.session_state[key] = as_date(days[min(max(i, 0), len(days) - 1)])


cur = snap(st.session_state[key])
c1, c2, c3, c4 = st.columns([0.95, 1.55, 0.95, 5], vertical_alignment="bottom")
c1.button("前一日", icon=":material/chevron_left:", key="mon_prev", on_click=shift, args=(-1,), width="stretch",
          disabled=cur == days[0])
picked = c2.date_input("交易日（盘后舆情 → 次日预判）", key=key, min_value=as_date(days[0]),
                       max_value=as_date(days[-1]), format="YYYY-MM-DD")
c3.button("后一日", icon=":material/chevron_right:", icon_position="right", key="mon_next", on_click=shift, args=(1,),
          width="stretch", disabled=cur == days[-1])
d = snap(picked)
i = days.index(d)
nd = days[i + 1] if i + 1 < len(days) else None
news_files = sorted(news.FLASH_DIR.glob("*.parquet"))
res = prediction_for(d, nd, m_lad, m_tm, m_rules, max((mtime(p) for p in news_files), default=0.0))
pred, msgs, events, hot, pop = res["pred"], res["msgs"], res["events"], res["hot"], res["pop"]
start, end = res["window"]
last_msg = msgs["time"].max() if len(msgs) else None
c4.caption(f"盘后窗口 {start:%m-%d %H:%M} ~ {end:%m-%d %H:%M}"
           + ("" if nd else "（下一交易日开盘前）") + f"：快讯 {len(msgs)} 条（重要 {int(msgs['important'].sum()) if len(msgs) else 0}）"
           + (f"，最新 {last_msg:%m-%d %H:%M}" if last_msg is not None else "")
           + ("" if nd else "。每日 15:45、19:30、22:30、次日 8:50 自动更新"))

day_lad = lad[lad["date"] == d]
up = day_lad[day_lad["is_up"]].assign(theme=themes[day_lad.index][day_lad["is_up"]])

tab_pred, tab_reason, tab_news, tab_rules = st.tabs(["明日预判", "涨停原因", "盘后舆情", "消息规则"])

# ================================================================ 明日预判
with tab_pred:
    nd_txt = fmt_day(nd) if nd else "下一交易日"
    st.caption(f"「爆发」= {nd_txt}该题材涨停 ≥ {TARGET_N} 家。基础概率由今日量价（涨停家数、昨日家数、最高板、连板、"
               "10 点前涨停、连续活跃天数、市场涨停数）按历史回测校准；催化分来自盘后消息（按「消息规则」打分）、"
               "事件日历、问财热门首次上榜、人气榜，未经回测，权重可调；调整后概率 = 二者合成。")
    mx = explode_themes(msgs[msgs["weight"] != 0]) if len(msgs) else pd.DataFrame(columns=["theme"])
    leaders = (up.sort_values(["streak", "first_sec"], ascending=[False, True]).groupby("theme")
               .apply(lambda g: "、".join(f"{n}{'' if s == 1 else f'{s}板'}" for n, s in zip(g["name"].head(3), g["streak"].head(3))),
                      include_groups=False))
    cat_txt = (mx.groupby("theme")["category"].apply(lambda s: " · ".join(f"{k}{v}" for k, v in Counter(s).most_common(4)))
               if len(mx) else pd.Series(dtype=str))
    top_msg = (mx.assign(a=mx["w_theme"].abs()).sort_values(["a", "important"], ascending=False).groupby("theme")["title"].first()
               if len(mx) else pd.Series(dtype=str))   # 关键消息：对题材催化最大的一条（个股公告已折算）
    ev = explode_themes(events) if len(events) else pd.DataFrame(columns=["theme", "title", "date"])
    ev_txt = ev.sort_values("date").groupby("theme")["title"].first() if len(ev) else pd.Series(dtype=str)
    show_all = st.toggle("显示全部题材", value=False, key="mon_all")
    t = pred if show_all else pred[(pred["n_up"] > 0) | (pred["cat_pts"] != 0)].head(20)
    table = pd.DataFrame({
        "题材": t["theme"], "调整后概率": (t["p_adj"] * 100).round(0), "基础概率": (t["p_base"] * 100).round(0),
        "催化分": t["cat"].round(2), "今日涨停": t["n_up"].astype(int), "最高板": t["max_h"].astype(int),
        "主要催化": t["theme"].map(cat_txt).fillna(""), "代表股": t["theme"].map(leaders).fillna(""),
        "关键消息": t["theme"].map(top_msg).fillna(""), "日历事件": t["theme"].map(ev_txt).fillna(""),
        "昨日涨停": t["n_prev"].astype(int), "连板": t["n_lb"].astype(int), "10点前": t["early"].astype(int),
        "活跃天数": t["hot_days"].astype(int), "消息分": t["news_pts"].round(1), "事件分": t["cal_pts"].round(1),
        "热度分": (t["hot_pts"] + t["pop_pts"]).round(1),
    })
    if nd is not None:
        table.insert(1, f"{fmt_day(nd)[5:]}实际", t["theme"].map(panel[panel["date"] == nd].set_index("theme")["n_up"]).fillna(0).astype(int).to_numpy())
    pc = st.column_config.ProgressColumn
    st.dataframe(table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 760), column_config={
        "调整后概率": pc("调整后概率", format="%.0f%%", min_value=0, max_value=100, help="基础概率 + 催化分合成"),
        "基础概率": pc("基础概率", format="%.0f%%", min_value=0, max_value=100, help="只看量价，历史回测校准"),
        "消息分": st.column_config.NumberColumn(help="盘后消息高出该题材常态的点数（个股公告折算 0.3，利空为负）"),
        "催化分": st.column_config.NumberColumn(help="sign·ln(1+|消息分+事件分+热度分|)"),
        "主要催化": st.column_config.TextColumn(width="medium"), "关键消息": st.column_config.TextColumn(width="large"),
        "代表股": st.column_config.TextColumn(width="medium"), "日历事件": st.column_config.TextColumn(width="medium"),
    })
    new = pred[(pred["n_up"] <= 1) & (pred["cat_pts"] >= 2)].head(10)
    if len(new):
        st.markdown("**潜在新题材**（今日涨停 ≤ 1 家，但盘后催化较强）："
                    + "；".join(f"{r.theme}（催化 {r.cat_pts:.1f}：{cat_txt.get(r.theme, '日历/热度')}）" for r in new.itertuples()))
    with st.expander("模型说明、回测与预判复盘"):
        m = model["metrics"]
        st.markdown(
            f"基础模型：逻辑回归，样本内 ≤ {fmt_day(m['split'])}（{m['n_train']} 个题材日），样本外 {m['n_test']} 个题材日"
            f"（爆发比例 {m['pos_rate_test']:.1%}）。样本外 AUC **{m['auc']:.3f}**（基线“按今日涨停家数排序” {m['auc_baseline']:.3f}）；"
            f"每日前 3 个题材的爆发命中率 **{m['top3']['precision']:.1%}**（基线 {m['top3_baseline']['precision']:.1%}），"
            f"次日最强题材落在前 3 的比例 {m['top3']['top_hit']:.1%}（基线 {m['top3_baseline']['top_hit']:.1%}）。"
            "量价模型只比“今天强的明天继续强”略好，增量主要要靠盘后催化——下面的复盘用来检验催化分是否有效。")
        coef = pd.DataFrame({"特征": [FEATURES[f] for f in model["features"]],
                             "标准化系数": np.round(model["coef"][1:], 3)})
        st.dataframe(coef, hide_index=True, width="content")
        rv = review_predictions(panel)
        if len(rv):
            mean = rv[["综合分", "基础概率", "今日涨停家数"]].mean()
            st.markdown(f"**预判复盘**（{len(rv)} 个交易日，其中回溯 {int(rv['retro'].sum())} 天）：前 3 个题材次日爆发的比例 — "
                        f"综合分 **{mean['综合分']:.0%}**、只看基础概率 {mean['基础概率']:.0%}、按今日涨停家数 {mean['今日涨停家数']:.0%}。"
                        "回溯 = 用当时盘后到次日开盘前的快讯补算（当时的热门、人气快照取不到）。")
            st.dataframe(rv.sort_values("date", ascending=False).rename(columns={"date": "交易日", "retro": "回溯"}),
                         hide_index=True, width="stretch")

# ================================================================ 涨停原因
with tab_reason:
    zt = news.load_ztfx(d)
    if zt.empty:
        st.info(f"{fmt_day(d)} 的涨停分析还没下载：运行 `uv run python scripts/download_news.py`（每日自动更新会下载）。")
    else:
        info = up.merge(zt[["code", "reason_revealed"]], on="code", how="left")
        info["items"] = [parse_reason(r, d, rules) for r in info["reason_revealed"]]
        info["fresh"] = [any(it.fresh for it in its) for its in info["items"]]
        n_fresh = int(info["fresh"].sum())
        src = Counter(k for its in info["items"] for k in {it.src_kind for it in its if it.section == "公司"})
        st.caption(f"{fmt_day(d)} 涨停 {len(info)} 只：当日催化（引用 {3} 天内的消息）{n_fresh} 只、题材属性（只有更早的背景信息）"
                   f"{len(info) - n_fresh} 只。公司原因来源：" + " · ".join(f"{k} {src[k]}" for k in SRC_ORDER if src.get(k))
                   + "。来源为文本识别（同花顺涨停分析“据……”），传闻大V 只能识别网传、据悉一类说法。")
        order = info.groupby("theme").agg(n=("code", "size"), h=("streak", "max")).sort_values(["n", "h"], ascending=False)
        for th, row in order.iterrows():
            g = info[info["theme"] == th].sort_values(["streak", "first_sec"], ascending=[False, True])
            fresh = int(g["fresh"].sum())
            boards = Counter(int(s) for s in g["streak"])
            btxt = "、".join(f"{'首板' if h == 1 else f'{h}板'} {n}" for h, n in sorted(boards.items(), reverse=True))
            with st.expander(f"{th} · 涨停 {len(g)} 只（{btxt}）· 当日催化 {fresh} 只", expanded=bool(row["n"] >= 5)):
                ind = Counter()
                ind_meta = {}
                for its in g["items"]:
                    for it in its:
                        if it.section == "行业":
                            k = it.text[:60]
                            ind[k] += 1
                            ind_meta.setdefault(k, it)
                parts = ['<div class="mon">']
                if ind:
                    parts.append("<b>行业原因（题材驱动）</b>")
                    for k, n in ind.most_common(6):
                        it = ind_meta[k]
                        tags = chips([it.src_kind, *it.catalysts]) + chips(["当日催化"] if it.fresh else [], "pos")
                        parts.append(f'<div class="item">{esc(it.text[:160])}<span class="n">{n} 只引用</span> {tags}</div>')
                srcs = Counter(k for its in g["items"] for k in {it.src_kind for it in its if it.section == "公司"})
                if srcs:
                    parts.append("<b>公司原因来源</b>：" + " · ".join(f"{k} {srcs[k]}" for k in SRC_ORDER if srcs.get(k)))
                parts.append("</div>")
                st.html("".join(parts))
                rows = []
                for r in g.itertuples():
                    comp = [it for it in r.items if it.section == "公司"]
                    rows.append({"名称": r.name, "板": "首板" if r.streak == 1 else f"{r.streak}板",
                                 "首次涨停": r.first_time if isinstance(r.first_time, str) else "",
                                 "涨停原因类别": r.reason if isinstance(r.reason, str) else "",
                                 "当日催化": "✓" if r.fresh else "",
                                 "来源": "、".join(dict.fromkeys(it.src_kind for it in comp)),
                                 "公司原因（首条）": comp[0].text[:120] if comp else ""})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                             column_config={"公司原因（首条）": st.column_config.TextColumn(width="large")})

# ================================================================ 盘后舆情
with tab_news:
    if msgs.empty:
        st.info("盘后窗口内还没有快讯。")
    else:
        f1, f2, f3, f4 = st.columns([2.2, 2.2, 1.2, 2], vertical_alignment="bottom")
        cat_opts = [c.name for c in rules.categories] + ["行情复盘", "未分类"]
        sel_cat = f1.multiselect("消息类别", cat_opts, key="mon_cat", placeholder="全部类别")
        th_cnt = Counter(t for ts in msgs["themes"] for t in ts)
        sel_th = f2.multiselect("题材", sorted(th_cnt, key=lambda x: -th_cnt[x]), key="mon_theme",
                                format_func=lambda x: f"{x} {th_cnt[x]}", placeholder="全部题材")
        only_imp = f3.toggle("只看重要", key="mon_imp")
        kw = f4.text_input("搜索", key="mon_kw", placeholder="标题或摘要包含…")
        v = msgs
        if sel_cat:
            v = v[v["category"].fillna("未分类").isin(sel_cat)]
        if sel_th:
            v = v[v["themes"].map(lambda ts: any(t in sel_th for t in ts))]
        if only_imp:
            v = v[v["important"]]
        if kw:
            v = v[v["title"].str.contains(kw, regex=False) | v["digest"].str.contains(kw, regex=False)]
        st.caption(f"显示 {len(v)} / {len(msgs)} 条（按时间倒序，最多 300 条）。类别与权重按「消息规则」，题材 = 文本关键词 + 关联个股的题材。")
        parts = ['<div class="mon">']
        for r in v.head(300).itertuples():
            cat = ([f"{r.category} {r.weight:+.1f}" if r.weight else r.category] if isinstance(r.category, str) else [])
            if isinstance(r.category, str) and r.weight and r.level == "个股":
                cat[0] += f"（个股×{rules.company_factor:g}）"
            names = list(r.names) if isinstance(r.names, (list, np.ndarray)) else []
            parts.append(
                f'<div class="msg"><div class="hd"><span class="tm">{r.time:%m-%d %H:%M}</span>'
                f'<a href="{esc(r.url)}" target="_blank">{esc(r.title)}</a></div>'
                f'<div class="dg">{esc(r.digest[:160])}{"…" if len(r.digest) > 160 else ""}</div>'
                f'<div class="tg">{chips(["重要"] if r.important else [], "imp")}'
                f'{chips(cat, "neg" if r.weight < 0 else ("pos" if r.weight > 0 else ""))}{chips(r.themes)}'
                f'{chips(names[:6])}</div></div>')
        parts.append("</div>")
        st.html("".join(parts))
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**问财热门概念 / 行业**" + (f"（快照 {hot['fetched_at'].iloc[0]:%m-%d %H:%M}）" if hot is not None and len(hot) else ""))
        if hot is not None and len(hot):
            th_ = Themer(tm)
            st.dataframe(pd.DataFrame({"类型": hot["kind"], "名称": hot["name"], "热度": hot["heat"].round(0),
                                       "涨停": hot["tag"].fillna(""), "上榜": hot["hot_tag"].fillna(""),
                                       "对应题材": hot["name"].map(th_.of_block).fillna("")}),
                         hide_index=True, width="stretch", height=420)
        else:
            st.caption("该日没有快照（只能在当天运行时取到）。")
    with h2:
        st.markdown("**人气榜前 100 · 排名上升最多**" + (f"（快照 {pop['fetched_at'].iloc[0]:%m-%d %H:%M}）" if pop is not None and len(pop) else ""))
        if pop is not None and len(pop):
            names = pd.read_parquet(news.RAW_DIR / "securities.parquet", columns=["full_code", "name"]).set_index("full_code")["name"]
            pp = pop.assign(name=pop["code"].map(names), theme=pop["code"].map(res["code_theme"]))
            pp = pp.sort_values("rank_change", ascending=False).head(30)
            st.dataframe(pd.DataFrame({"名称": pp["name"], "排名": pp["rank"], "上升": pp["rank_change"],
                                       "题材": pp["theme"].fillna("")}), hide_index=True, width="stretch", height=420)
        else:
            st.caption("该日没有快照。")
    st.markdown("**事件日历**（下一交易日起 3 天内）")
    if len(events):
        st.dataframe(pd.DataFrame({"日期": events["date"].map(fmt_day), "事件": events["title"],
                                   "关联板块": events["blocks"].map(lambda b: "、".join(b) if isinstance(b, (list, np.ndarray)) else ""),
                                   "对应题材": events["themes"].map("、".join)}),
                     hide_index=True, width="stretch")
    else:
        st.caption("没有日历事件。")

# ================================================================ 消息规则
with tab_rules:
    hits = Counter(msgs["category"].dropna()) if len(msgs) else Counter()
    st.caption(f"消息定义：`{RULES_PATH.relative_to(PROJECT_ROOT)}`，修改后保存、刷新页面即生效。一条消息命中多个类别时按权重绝对值最大的计分；"
               f"同花顺标为“重要”的乘以 {rules.important_boost}。综合分 = logit(基础概率) + {rules.catalyst_beta} × 催化分；"
               f"事件日历每条 +{rules.calendar_weight}（下一交易日，之后 3 天内减半），问财热门首次上榜 +{rules.hot_new_weight}，"
               f"人气榜每只 +{rules.popularity_weight}。")
    st.dataframe(pd.DataFrame({"类别": [c.name for c in rules.categories], "方向": [c.kind for c in rules.categories],
                               "权重": [c.weight for c in rules.categories],
                               "当日盘后命中": [hits.get(c.name, 0) for c in rules.categories],
                               "关键词": ["、".join(c.keywords) for c in rules.categories]}),
                 hide_index=True, width="stretch", column_config={"关键词": st.column_config.TextColumn(width="large")})
    raw_rules = __import__("yaml").safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}
    st.markdown("**行情复盘（不计分）**：标题含 " + "、".join(raw_rules.get("ignore_titles") or []) + " 的消息只显示、不算催化。")
    st.markdown("**新闻归题材时不用的泛化词**：" + "、".join(raw_rules.get("theme_stopwords") or [])
                + "（只影响快讯 → 题材，不影响涨停原因标签的归类）。")
    st.markdown("**涨停原因来源分类**（静态分析里“据……”的来源，从上到下匹配）")
    st.dataframe(pd.DataFrame({"来源": [k for k, _ in rules.reason_sources],
                               "关键词": [p.pattern.replace("\\", "").replace("|", "、") for _, p in rules.reason_sources]}),
                 hide_index=True, width="stretch")
