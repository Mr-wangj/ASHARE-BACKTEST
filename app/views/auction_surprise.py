"""看板页面：竞价超预期 —— 每天 9:27 出结果（竞价表 9:25～9:27 落盘，当日打开自动刷新）。

上涨超预期（三类）：
1. 一字板（竞价涨停，标注第几板）；
2. 竞价高开（昨日未涨停，高开 ≥ 0.3 个涨停）且板块效应（同题材 ≥ 3 只竞价高开 ≥ 0.2 个涨停）；
3. 连板竞价弱转强（昨日涨停，弱转强口径 或 竞价超预期 ≥ 0.5 个涨停）。
下跌超预期：昨日涨停今日竞价跌停或跌幅 ≥ 5%。
每类给出所属板块的联动个股（同题材今日竞价高开的前几只）与昨收后（昨日 15:00 → 今日 9:25）
快讯中关联到该股的消息标题。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from ashare_bt.config import RAW_DIR
from ashare_bt.data import news
from ashare_bt.live.context import trading_days
from ashare_bt.live.feed import LIVE_DIR
from views._auction_common import load_day, limit_of


@st.cache_data(ttl=300)
def overnight_news(day: int) -> dict[str, str]:
    """昨收后（昨日 15:00 → 今日 9:25）快讯按个股关联：full_code → 消息标题（最多 2 条）。"""
    cal = trading_days()
    if day not in cal:
        return {}
    i = cal.index(day)
    pd_ = cal[i - 1] if i > 0 else day
    start = datetime.strptime(str(pd_), "%Y%m%d").replace(hour=15)
    end = datetime.strptime(str(day), "%Y%m%d").replace(hour=9, minute=25)
    try:
        df = news.load_flash(start, end)
    except Exception:                       # noqa: BLE001  快讯缺失不影响页面
        return {}
    if not len(df):
        return {}
    sec = pd.read_parquet(RAW_DIR / "securities.parquet", columns=["full_code", "name"]).set_index("name")
    m: dict[str, list[str]] = {}
    for r in df.itertuples():
        text = f"{pd.Timestamp(r.time):%H:%M} {str(r.title)[:46]}"
        codes = set(np.asarray(r.codes).tolist())
        for n in np.asarray(r.names).tolist():
            if n in sec.index:
                codes.add(sec.loc[n, "full_code"])
        for c in codes:
            m.setdefault(c, []).append(text)
    return {c: "；".join(v[:2]) for c, v in m.items()}


@st.cache_data(ttl=600)
def _blocks() -> dict[str, list[tuple[str, frozenset]]]:
    """个股 → 所属同花顺板块 [(板块名, 成分集合)]（concepts.parquet 反查；无题材股的板块效应用）。
    与盘中监控同一套排除规则（剔除融资融券、沪深股通等非题材板块，成分 8～400 只）。"""
    import yaml

    from ashare_bt.live.store import RULES_PATH
    c = pd.read_parquet(LIVE_DIR / "concepts.parquet")
    kw = (yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}).get("sector", {})
    bad = "|".join(map(str, kw.get("exclude_keywords") or ["融资融券"]))
    c = c[~c["name"].str.contains(bad, regex=True)]
    c = c[c["n"].between(kw.get("min_size", 8), kw.get("max_size", 400))]
    out: dict[str, list[tuple[str, frozenset]]] = {}
    for r in c.itertuples():
        ms = frozenset(np.asarray(r.members).tolist())
        for m in ms:
            out.setdefault(m, []).append((r.name, ms))
    return out


st.title("竞价超预期")
days = sorted((int(p.name) for p in LIVE_DIR.glob("20*") if (p / "auction.parquet").exists()), reverse=True) \
    if LIVE_DIR.exists() else []
if not days:
    st.info("还没有竞价数据（实时监控须 9:25 前启动；竞价失败时收盘后会分时自愈补表）。")
    st.stop()
today = int(datetime.now().strftime("%Y%m%d"))
day = st.selectbox("交易日", days, key="sp_day",
                   format_func=lambda d: f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}"
                   + ("（今天，9:25～9:27 出数）" if d == today else ""))
cal = trading_days()
t = load_day(day, cal)
if t.empty:
    st.info("这一天没有竞价数据。")
    st.stop()
news_map = overnight_news(day)
t["limit_pct"] = t["code"].map(limit_of)
pn = t["pct"] / t["limit_pct"]
t["pct_norm"] = pn
t["msg"] = t["code"].map(news_map).fillna("")
y = t[t["y_up"]]

running = day == today and datetime.now().hour < 15


def show(rows: pd.DataFrame, note: str, board_col: bool = True) -> None:
    """上涨 / 下跌表：带板块联动与昨收后消息。eff_block / eff_link 列存在时优先用（②类板块兜底）。"""
    if not len(rows):
        st.markdown(f"**{note}**：无。")
        return
    hot = t[t["pct_norm"] >= 0.2].sort_values("pct", ascending=False)
    if "eff_link" in rows:
        members = rows["eff_link"]
    else:
        members = {r.code: "、".join(f"{h.name} {h.pct:+.1%}" for h in
                                     hot[(hot["theme"] == r.theme) & (hot["code"] != r.code)].head(4).itertuples())
                   for r in rows.itertuples()}
    block_col_name = "板块" if "eff_block" in rows else "题材"
    st.markdown(f"**{note}**（{len(rows)} 只）")
    st.dataframe(pd.DataFrame({
        "名称": rows["name"], "板数": rows["板数"] if "板数" in rows else rows["y_streak"],
        "竞价涨幅%": (rows["pct"] * 100).round(2), "竞价额(万)": (rows["amount"] / 1e4).round(0),
        "竞价额/昨日%": (rows["ratio"] * 100).round(1),
        "封单(万)": (rows["seal_amt"].fillna(0) / 1e4).round(0) if board_col else "",
        block_col_name: rows["eff_block"] if "eff_block" in rows else rows["theme"],
        "板块联动（竞价高开）": rows["code"].map(members) if isinstance(members, dict) else members,
        "昨收后消息": rows["msg"],
    }), hide_index=True, width="stretch", height=min(38 + 35 * len(rows), 420))


@st.fragment(run_every=5 if running else None)
def body() -> None:
    boards = y.apply(lambda r: f"{int(r['y_streak']) + 1}板" if r["y_up"] else "首板", axis=1)
    yb = y.assign(板数=boards)

    # ① 一字板（竞价涨停，含首板 / 连板）
    one = yb[yb["at_limit"]].sort_values(["y_streak", "amount"], ascending=[False, False])
    show(one, "① 一字板（竞价涨停）")

    # ② 竞价高开 × 板块效应（昨日未涨停：高开 ≥ 0.3 个涨停；板块效应 = 主题材 ≥ 3 只高开 ≥ 0.2 个涨停，
    #    近期没涨停过、无题材的股票用同花顺板块成分判定——如易点天下：AI应用 / AIGC 概念）
    hot_cnt = t[t["pct_norm"] >= 0.2].groupby("theme")["code"].size().to_dict()
    cand = t[~t["y_up"] & (t["pct_norm"] >= 0.3)]
    blocks = _blocks()
    hot_codes = frozenset(t.loc[t["pct_norm"] >= 0.2, "code"])
    pick = []
    for r in cand.itertuples():
        if isinstance(r.theme, str) and r.theme and hot_cnt.get(r.theme, 0) >= 3:
            pick.append([r, r.theme, None])
            continue
        best = None
        for bname, ms in blocks.get(r.code, []):        # 无题材：任一所属板块内 ≥ 3 只高开
            inter = ms & hot_codes
            if len(inter) >= 3 and (best is None or len(inter) > len(best[2])):
                best = [r, bname, inter]
        if best:
            pick.append(best)
    surge = pd.DataFrame([p[0] for p in pick])
    surge["板数"] = "-"
    surge["eff_block"] = [p[1] for p in pick]
    if len(surge):
        hot = t[t["pct_norm"] >= 0.2].sort_values("pct", ascending=False)
        links = []
        for p in pick:
            pool = hot[hot["code"].isin(p[2] or set()) & (hot["code"] != p[0].code)] if p[2] is not None else \
                hot[(hot["theme"] == p[1]) & (hot["code"] != p[0].code)]
            links.append("、".join(f"{h.name} {h.pct:+.1%}" for h in pool.head(4).itertuples()))
        surge["eff_link"] = links
    show(surge.sort_values("pct", ascending=False), "② 竞价高开 × 板块效应（昨日未涨停）")

    # ③ 连板竞价弱转强（昨日涨停：弱转强 或 竞价超预期 ≥ 0.5 个涨停）
    wts = yb[(yb["wts"] | yb["beyond"])].sort_values("auc_vs_yauc", ascending=False, na_position="last")
    show(wts, "③ 连板竞价弱转强（昨日涨停）")

    # ④ 下跌超预期（昨日涨停：竞价跌停 或 跌幅 ≥ 5%）
    dn = yb[(yb["pct"] <= -0.05) | (yb["pct"] <= -yb["limit_pct"] + 0.002)].sort_values("pct")
    show(dn, "④ 下跌超预期（昨日涨停：跌停 / ≤ -5%）", board_col=False)


body()
st.caption("竞价表 9:25～9:27 落盘后本页当日每 5 秒自动刷新；昨收后消息 = 昨日 15:00 到今日 9:25 的同花顺快讯"
           "（关联个股），高开 = 竞价涨幅 ÷ 涨跌幅限制。上涨①②③与下跌④的口径见页首说明。"
           "竞价失败的日子（如 11-05 上午认证失效）收盘后自动分时自愈补表。")
