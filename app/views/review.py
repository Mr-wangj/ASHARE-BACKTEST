"""看板页面：复盘与预期 —— 上半页回顾今日盘面异动，下半页推演明日。

数据（全部本地）：
- data/features/market_mood.parquet       情绪概况（run_backtest 生成，收盘后更新）
- data/features/limitup_ladder.parquet     涨停 / 炸板 / 断板明细（含首次涨停时间、涨停原因、竞价涨幅）
- data/features/theme_predictions/{日期}.parquet   明日题材预判（题材涨停 ≥ 3 家的概率，download_news 生成）
- data/live/{日期}/alerts.jsonl            盘中异动预警流（live_monitor，仅有记录的交易日）

明日弱转强观察池：养家心法口径 —— 今日 13:00 后才封板的尾盘弱板（弱），以及今日炸板股
（更弱），明日竞价若高开 ≥ 0.2 个涨停且放量（见「竞价弱转强」页口径），即弱转强。
观察池按所属题材的明日预判概率排序：题材要晋级，个股的弱转强才有土壤。
"""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from ashare_bt.config import FEATURE_DIR, RAW_DIR
from ashare_bt.live.feed import LIVE_DIR
from ashare_bt.themes import categories, load_theme_map, main_category, split_tags

tm = load_theme_map()
CSS = """<style>
.lv .a { display: flex; gap: 10px; align-items: baseline; padding: 5px 2px; border-bottom: 1px solid rgba(128,128,128,.16); }
.lv .tm { flex: 0 0 64px; font-size: 12.5px; opacity: .65; font-variant-numeric: tabular-nums; }
.lv .lv { flex: 0 0 auto; font-size: 11.5px; padding: 0 6px; border-radius: 4px; line-height: 18px; }
.lv .lv.h { background: #fee2e2; color: #b91c1c; } .lv .lv.m { background: #fef3c7; color: #92400e; }
.lv .lv.l { background: rgba(128,128,128,.15); }
.lv .k { flex: 0 0 78px; font-size: 12px; opacity: .8; }
.lv .t { font-weight: 600; font-size: 13.5px; } .lv .d { font-size: 12.5px; opacity: .72; margin-left: 6px; }
</style>"""


def esc(s) -> str:
    return html.escape("" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s))


@st.cache_data(ttl=600)
def mood_day(day: int) -> pd.Series | None:
    m = pd.read_parquet(FEATURE_DIR / "market_mood.parquet")
    m = m[m["date"] == day]
    return m.iloc[0] if len(m) else None


@st.cache_data(ttl=600)
def mood_days(days: list[int]) -> pd.DataFrame:
    m = pd.read_parquet(FEATURE_DIR / "market_mood.parquet", columns=["date", "zt", "max_h", "zb_rate", "temp"])
    return m[m["date"].isin(days)].set_index("date").sort_index()


@st.cache_data(ttl=600)
def ladder_day(day: int) -> pd.DataFrame:
    d = pd.read_parquet(FEATURE_DIR / "limitup_ladder.parquet")
    d = d[d["date"] == day].copy()
    if not len(d):
        return d
    d["cats"] = [categories(split_tags(r), tm) if isinstance(r, str) else [] for r in d["reason"]]
    d["theme"] = [main_category(c, tm) or "其他" for c in d["cats"]]
    d["prv_theme"] = [main_category(categories(split_tags(r), tm), tm) if isinstance(r, str) else ""
                      for r in d["prv_reason"]]
    d["boards"] = np.where(d["kind"] == "涨停", d["streak"], d["prv_streak"])
    try:                                    # 行业列：首次全市场拉取约几秒，之后读缓存
        from ashare_bt.data.industry import industry_of
        ind = industry_of(d["code"].tolist())
        d["ind1"], d["ind2"] = ind["ind1"].to_numpy(), ind["ind2"].to_numpy()
    except Exception as exc:                # noqa: BLE001  行业拉取失败不影响页面
        d["ind1"] = d["ind2"] = None
        st.caption(f"行业映射拉取失败（{str(exc)[:60]}），行业列为空。")
    return d


@st.cache_data(ttl=600)
def predict_day(day: int) -> pd.DataFrame:
    p = FEATURE_DIR / "theme_predictions" / f"{day}.parquet"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    d = d[~d.get("retro", False)] if "retro" in d else d
    return d.sort_values("p_adj", ascending=False)


SEG_BINS = [0, 570, 600, 630, 660, 690, 780, 810, 840, 870, 900]
SEG_LABELS = ["竞价", "9:30-10:00", "10:00-10:30", "10:30-11:00", "11:00-11:30", "午休",
              "13:00-13:30", "13:30-14:00", "14:00-14:30", "14:30-15:00"]
BAD_WORDS = ["监管", "立案", "核查", "停牌", "重点监控", "严重异动", "减持", "问询", "处罚", "风险警示",
             "冻结", "离婚", "退市", "终止上市", "澄清", "业绩预亏", "亏损"]


@st.cache_data(ttl=600)
def flash_after(day: int) -> pd.DataFrame:
    """当日 15:00 后的快讯（盘后窗口，事件假设用）。"""
    p = RAW_DIR / "news" / "flash" / f"{day}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["dt"] = pd.to_datetime(df["time"])
    df = df[df["dt"].dt.strftime("%H%M") >= "1500"].sort_values("dt")
    return df[["dt", "title", "digest", "codes", "names", "important"]]


SH_INDEX = "USHI1A0001"          # 上证指数
MOVING_MIN = 3                   # 题材异动门槛：当日至少 3 只涨停才算异动


@st.cache_data(ttl=600)
def index_intraday(day: int, prev_close: float) -> pd.DataFrame:
    """上证指数分时（涨跌幅%），缓存 data/live/index/。"""
    from ashare_bt.live.feed import Feed
    df = Feed().index_intraday(SH_INDEX, day)
    if df is None or not len(df):
        return pd.DataFrame()
    out = df.copy()
    out["pct"] = (out["price"] / prev_close - 1) * 100
    return out


def theme_events(hot: pd.DataFrame) -> pd.DataFrame:
    """题材异动事件：当日涨停 ≥ MOVING_MIN 只的题材，事件时点 = 题材内第 3 只封板时间（成势点）。"""
    if not len(hot):
        return pd.DataFrame()
    h = hot[hot["first_time"].notna()].sort_values(["theme", "first_time"])
    g = h.groupby("theme")
    n = g.size()
    ev = g.nth(MOVING_MIN - 1).reset_index()          # 第 3 只（不足 3 只的题材为 NaN）
    ev = ev[ev["theme"].isin(n[n >= MOVING_MIN].index)]
    d = str(hot["date"].iloc[0])
    t = pd.to_datetime(d + " " + ev["first_time"].astype(str), format="%Y%m%d %H:%M:%S", errors="coerce")
    srt = hot.sort_values(["theme", "boards", "first_time"], ascending=[True, False, True], na_position="last")
    top3 = srt.groupby("theme", group_keys=False).apply(
        lambda x: "、".join(f"{r.name}({int(r.boards)}板 {r.first_time if isinstance(r.first_time, str) else '竞价'})"
                           for r in x.head(3).itertuples()), include_groups=False)
    return pd.DataFrame({"time": t, "theme": ev["theme"].to_numpy(), "n": n[ev["theme"]].to_numpy(),
                         "top3": ev["theme"].map(top3).fillna("").to_numpy()})


def seg_of(first_time: pd.Series) -> pd.Series:
    m = first_time.fillna("09:30:00").str.slice(0, 5).map(lambda t: int(t[:2]) * 60 + int(t[3:5]))
    return pd.cut(m, bins=SEG_BINS, labels=SEG_LABELS, right=True, include_lowest=False)


def rotation(hot: pd.DataFrame) -> pd.DataFrame:
    """题材轮动时间轴：每个时段里各题材新封涨停的只数（含题材内最高板与代表股）。"""
    if not len(hot):
        return pd.DataFrame()
    s = seg_of(hot["first_time"])
    g = hot.assign(seg=s).groupby(["seg", "theme"], observed=True)
    out = g.agg(新涨停=("name", "size"), 最高板=("boards", "max")).reset_index()
    top = g.apply(lambda x: x.sort_values(["boards", "first_time"], ascending=[False, True]).iloc[0],
                  include_groups=False)[["name", "first_time", "boards"]].reset_index()
    out["代表"] = top["name"] + "（" + top["first_time"].astype(str) + "）"
    order = {lab: k for k, lab in enumerate(SEG_LABELS)}
    return out.sort_values(["seg", "新涨停", "最高板"], ascending=[True, False, False],
                           key=lambda c: c.map(order) if c.name == "seg" else c)


def theme_six(hot: pd.DataFrame) -> pd.DataFrame:
    """每题材最强 6 只：20cm 3 + 10cm 3（北交所并入 20cm 桶），强度 = 连板高度 > 涨停时间。"""
    if not len(hot):
        return pd.DataFrame()
    h = hot.copy()
    h["cm"] = np.where(h["board"] == "main", 10, 20)
    h = h.sort_values(["theme", "boards", "first_time"], ascending=[True, False, True], na_position="last")
    sel = h.groupby(["theme", "cm"]).head(3)
    sel = sel.sort_values(["theme", "cm", "boards", "first_time"], ascending=[True, False, False, True])
    return sel


def assumptions(lad: pd.DataFrame, pred: pd.DataFrame, flash: pd.DataFrame) -> pd.DataFrame:
    """明日假设（推演，不是预测）：走强 / 退潮 / 盘后利空事件，均给确认信号。"""
    rows: list[dict] = []
    hot = lad[lad["kind"] == "涨停"] if len(lad) else pd.DataFrame()
    six = theme_six(hot)
    for r in pred[(pred["p_adj"] >= 0.7) & (pred["n_up"].fillna(0) >= 3)].head(6).itertuples():
        core = six[six["theme"] == r.theme]["name"].tolist()[:2]
        rows.append({"类别": "走强", "事件 / 依据": f"{r.theme}：今日涨停 {int(r.n_up)} 只（最高 {int(r.max_h)} 板），"
                    f"明日晋级概率 {r.p_adj:.0%}（催化 {r.cat_pts:.0f}）", "涉及股票": "、".join(core) or "-",
                    "推演": f"{r.theme} 明日大概率高开晋级；确认：竞价 ≥ 2 只高开超 0.2 个涨停。"
                    "若核心股竞价核按钮（低开 ≥ 3%）则假设作废，题材进入分歧"})
    down = lad[lad["kind"].isin(["炸板", "断板"]) & (lad["boards"] >= 2)] if len(lad) else pd.DataFrame()
    for r in down.sort_values("boards", ascending=False).head(8).itertuples():
        what = "炸板（触板未封）" if r.kind == "炸板" else "断板（未再涨停）"
        th = r.theme if r.theme and r.theme != "其他" else (r.prv_theme or "其他")
        rows.append({"类别": "退潮", "事件 / 依据": f"{r.name}（{int(r.boards)} 板高度股）今日{what}",
                    "涉及股票": str(r.name), "推演": f"{th}题材进入高位分歧：明日竞价弱 → 退潮确认；"
                    "按心法此时不接力高位，观察低位补涨龙（老龙头断板日的首板）"})
    if len(flash) and len(hot):
        up_codes, up_names = set(hot["code"]), set(hot["name"])
        for f in flash.itertuples():
            text = str(f.title) + str(f.digest or "")
            if not any(w in text for w in BAD_WORDS):
                continue
            hit = (set(np.asarray(f.codes).tolist()) & up_codes) | (set(np.asarray(f.names).tolist()) & up_names)
            hit |= {n for n in up_names if n in str(f.title)[:12]}          # 标题「XX：」前缀
            if not hit:
                continue
            hs = hot[hot["code"].isin(hit) | hot["name"].isin(hit)]
            for r in hs.itertuples():
                rows.append({"类别": "利空", "事件 / 依据": f"盘后快讯：{str(f.title)[:70]}",
                             "涉及股票": str(r.name),
                             "推演": f"{r.name}（{int(r.boards)} 板、{r.theme or '无题材'}）明日竞价核按钮概率大；"
                             "若是题材核心 / 高度股，题材情绪受损、可能整体退潮（参考 10-30 新华传媒进监管 → "
                             "AI 短剧次日退潮）"})
    return pd.DataFrame(rows, columns=["类别", "事件 / 依据", "涉及股票", "推演"])


st.title("复盘与预期")
all_days = pd.read_parquet(FEATURE_DIR / "market_mood.parquet", columns=["date"])["date"]
days = sorted(all_days.unique().astype(int).tolist(), reverse=True)
live_days = {int(p.name) for p in LIVE_DIR.glob("20*") if (p / "alerts.jsonl").exists()} if LIVE_DIR.exists() else set()
day = st.selectbox("交易日", days, key="rev_day", format_func=lambda d: f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}"
                   + ("（有盘中异动记录）" if d in live_days else ""))
md, lad, pred = mood_day(day), ladder_day(day), predict_day(day)
if md is None:
    st.warning("这一天没有数据。")
    st.stop()
today = int(datetime.now().strftime("%Y%m%d"))
tomorrow_note = "（明日即今日的下一交易日）" if day == days[0] else "（历史回看）"

# ============================================================ Part 1 今日盘面异动
st.header("今日盘面异动")
c = st.columns(8)
c[0].metric("涨停", int(md["zt"]))
c[1].metric("炸板", int(md["fb"]), f"炸板率 {md['zb_rate']:.0%}" if pd.notna(md["zb_rate"]) else None)
c[2].metric("最高板", int(md["max_h"]), md["top_names"] if pd.notna(md["top_names"]) else None)
c[3].metric("连板", int(md["lb"]))
c[4].metric("首板晋级率", f"{md['promo_fb']:.0%}")
c[5].metric("昨日涨停溢价", f"{md['prem_zt'] * 100:+.1f}%")
c[6].metric("大面 / 天地板", f"{int(md['damian'])} / {int(md['tiandi'])}")
c[7].metric("情绪温度", f"{md['temp']:.0f}", md["temp_level"])

hot = lad[lad["kind"] == "涨停"].copy() if len(lad) else pd.DataFrame()
cnt = hot.groupby("theme").size() if len(hot) else pd.Series(dtype=int)
big_themes = set(cnt[cnt >= MOVING_MIN].index)          # 异动题材门槛：≥ 3 只涨停

# 大盘分时 × 题材异动标注
sh = pd.read_parquet(RAW_DIR / "index_daily.parquet", filters=[("index", "==", "sh")], columns=["date", "close"])
prev = float(sh[sh["date"] < day]["close"].iloc[-1]) if (sh["date"] < day).any() else None
try:
    idp = index_intraday(day, prev) if prev else pd.DataFrame()
except Exception as exc:      # noqa: BLE001  网络或认证失败：降级为无图，不阻塞页面
    idp = pd.DataFrame()
    st.caption(f"指数分时拉取失败（{str(exc)[:60]}），跳过分时图。")
if len(idp):
    import altair as alt
    ev = theme_events(hot)
    idp = idp.copy()
    idp["hm"] = idp["time"].dt.strftime("%H:%M")                # 类目轴：分钟格，午休天然不存在
    if len(ev):
        m = pd.merge_asof(ev.sort_values("time"), idp[["time", "pct", "hm"]].sort_values("time"),
                          on="time", direction="backward", tolerance=pd.Timedelta("10min"))
        ev = m.dropna(subset=["pct"]).copy()
        ev["label"] = ev["theme"] + " " + ev["n"].astype(int).astype(str) + "只"
        ev["label_y"] = ev["pct"] + 0.08 + (np.arange(len(ev)) % 5) * 0.22   # 标签错开，避免重叠
    hm_order = idp["hm"].tolist()
    ticks = [h for h in hm_order if h.endswith((":00", ":30"))]
    X = alt.X("hm:N", sort=hm_order, title=None, axis=alt.Axis(values=ticks, labelAngle=0))
    Y = alt.Y("pct:Q", title="上证涨跌幅%", scale=alt.Scale(zero=False))
    line = alt.Chart(idp).mark_line(color="#1a6fc9", point=False).encode(x=X, y=Y)
    zero = alt.Chart(pd.DataFrame({"pct": [0]})).mark_rule(strokeDash=[4, 4], color="#999").encode(y="pct")
    charts = line + zero
    if len(ev):
        pts = alt.Chart(ev).mark_point(filled=True, size=130, color="#d62728").encode(
            x=alt.X("hm:N", sort=hm_order), y="pct:Q",
            tooltip=[alt.Tooltip("hm:N", title="成势时间"), "theme", "n",
                     alt.Tooltip("top3:N", title="强度前3（板数>首停时间）")])
        txt = alt.Chart(ev).mark_text(color="#b91c1c", fontSize=11).encode(
            x=alt.X("hm:N", sort=hm_order), y="label_y:Q", text="label")
        charts = charts + pts + txt
    st.altair_chart(charts.interactive().properties(height=380), width="stretch")
    st.caption(f"上证指数分时（相对昨收），红色标记 = 题材异动（当日 ≥ {MOVING_MIN} 只涨停，标注点为题材内第 3 只"
               "封板时间，即成势点）。缩放拖动可看细节。")
elif day == days[0]:
    st.info("拉不到当日指数分时（需登录同花顺；实时分时只保留近期）。可在数据控制台跑一次「盘中监控自检」"
            "后重试，或直接看下面的题材轮动时间轴。")

if len(lad):
    hot3 = hot[hot["theme"].isin(big_themes)] if len(hot) else hot
    rot = rotation(hot3)
    st.markdown(f"**题材轮动时间轴**（当日 ≥ {MOVING_MIN} 只涨停的异动题材，共 {len(big_themes)} 个："
                f"{len(hot) if len(hot) else 0} 只涨停股按时段 × 题材聚合；「代表」= 该时段题材内板数最高、"
                "最早封板的股票）")
    st.dataframe(pd.DataFrame({
        "时段": rot["seg"], "题材": rot["theme"], "新涨停": rot["新涨停"], "题材最高板": rot["最高板"],
        "代表": rot["代表"],
    }), hide_index=True, width="stretch", height=min(38 + 35 * len(rot), 420))

    six = theme_six(hot3)
    st.markdown(f"**异动题材最强 6 只**（{len(big_themes)} 个题材：20cm 3 只、"
                "10cm 3 只，北交所并入 20cm 桶；强度 = 连板高度优先、其次涨停时间早；行业为同花顺一 / 二级）")
    first = ~six.duplicated("theme")
    st.dataframe(pd.DataFrame({
        "题材": np.where(first, six["theme"], ""), "名称": six["name"],
        "一级行业": six["ind1"].fillna(""), "二级行业": six["ind2"].fillna(""),
        "板数": six["boards"], "首停": six["first_time"].fillna(""), "cm": six["cm"],
        "收盘涨幅%": (six["close_pct"] * 100).round(1), "涨停原因": six["reason"].fillna(""),
    }), hide_index=True, width="stretch", height=min(38 + 35 * len(six), 520),
        column_config={"涨停原因": st.column_config.TextColumn(width="medium")})

    with st.expander(f"涨停 / 炸板时间线（全部 {int((lad['kind'] == '涨停').sum())} 涨停 + "
                     f"{int((lad['kind'] == '炸板').sum())} 炸板，按首次涨停时间排序）"):
        z = lad[lad["kind"].isin(["涨停", "炸板"])].sort_values("first_time", na_position="last")
        st.dataframe(pd.DataFrame({
            "时间": z["first_time"].fillna(""), "名称": z["name"], "板数": z["boards"],
            "状态": z["kind"], "一级行业": z["ind1"].fillna(""), "二级行业": z["ind2"].fillna(""),
            "竞价涨幅%": (z["open_pct"] * 100).round(1), "收盘涨幅%": (z["close_pct"] * 100).round(1),
            "成交额(亿)": (z["amount"] / 1e8).round(2), "题材": z["theme"], "涨停原因": z["reason"].fillna(""),
        }), hide_index=True, width="stretch", height=min(38 + 35 * len(z), 520),
            column_config={"涨停原因": st.column_config.TextColumn(width="medium")})

# ============================================================ Part 2 明日预期
st.header(f"明日预期 {tomorrow_note}")
if len(pred):
    st.markdown(f"**题材晋级预期**（明日涨停 ≥ 3 家的概率，前 15；概率 = 基础模型 × 盘后催化调整）")
    p = pred.head(15)
    st.dataframe(pd.DataFrame({
        "题材": p["theme"], "明日爆发概率": (p["p_adj"] * 100).round(1), "基础概率": (p["p_base"] * 100).round(1),
        "催化分": p["cat_pts"].round(1), "今日涨停": p["n_up"].fillna(0).astype(int),
        "昨日涨停": p["n_prev"].fillna(0).astype(int), "题材最高板": p["max_h"].fillna(0).astype(int),
        "活跃天数": p["hot_days"].fillna(0).astype(int), "连板": p["n_lb"].fillna(0).astype(int),
    }), hide_index=True, width="stretch")
else:
    st.info("这一天没有明日预判（data/features/theme_predictions/{日期}.parquet，盘后 download_news 生成）。")

asm = assumptions(lad, pred, flash_after(day) if day == days[0] or day in live_days else pd.DataFrame())
if len(asm):
    st.markdown(f"**明日假设**（{len(asm)} 条推演：走强 / 退潮 / 盘后利空事件——假设需要次日竞价验证，不是预测）")
    for r in asm.itertuples():
        icon = {"走强": "🔴", "退潮": "🟢", "利空": "⚫"}.get(r.类别, "·")      # 红涨绿跌
        st.markdown(f"{icon} **{r.类别} · {r._2}**  \n"
                    f"<span style='opacity:.75;font-size:12.5px'>涉及：{esc(r.涉及股票)} —— {esc(r.推演)}</span>",
                    unsafe_allow_html=True)
else:
    st.markdown("**明日假设**：今日无明显的走强 / 退潮 / 盘后利空信号。")

if len(lad):
    watch = lad[(lad["kind"] == "涨停") & (lad["first_time"] >= "13:00:00")].copy()
    watch["类型"] = "尾盘弱板"
    zb = lad[lad["kind"] == "炸板"].copy()
    zb["类型"] = "今日炸板"
    watch = pd.concat([watch, zb])
    st.markdown(f"**明日弱转强观察池**（{len(watch)} 只：尾盘弱板 {int((watch['类型'] == '尾盘弱板').sum())}"
                f"（今日 13:00 后才封板）、今日炸板 {int((watch['类型'] == '今日炸板').sum())}；按所属题材的"
                "明日预判概率排序——题材要晋级，弱转强才有土壤）")
    if len(watch):
        pp = pred.set_index("theme")["p_adj"] if len(pred) else pd.Series(dtype=float)
        watch = watch.assign(theme_p=(watch["theme"].map(pp).fillna(0) * 100).round(1))
        w = watch.sort_values(["theme_p", "类型", "first_time"], ascending=[False, True, False], na_position="last")
        st.dataframe(pd.DataFrame({
            "名称": w["name"], "类型": w["类型"], "板数": w["boards"], "首停时间": w["first_time"].fillna(""),
            "今日收盘%": (w["close_pct"] * 100).round(1), "一级行业": w["ind1"].fillna(""),
            "题材": w["theme"], "题材明日概率%": w["theme_p"], "涨停原因": w["reason"].fillna(""),
        }), hide_index=True, width="stretch", height=min(38 + 35 * len(w), 460),
            column_config={"涨停原因": st.column_config.TextColumn(width="medium")})
        st.caption("弱转强口径见「竞价弱转强」页：明日竞价高开 ≥ 0.2 个涨停且竞价额 ≥ 昨日全天 5%。"
                   "观察池只是候选名单，竞价不达标就不是弱转强；尾盘板次日竞价核按钮（低开 ≥ 3%）同样常见。")
    else:
        st.caption("今日没有尾盘弱板和炸板股，明日弱转强观察池为空。")

# ============================================================ 风险信号
hist = mood_days(days[:6][::-1])
signals = []
if pd.notna(md["zb_rate"]) and md["zb_rate"] >= 0.4:
    signals.append(f"炸板率 {md['zb_rate']:.0%} ≥ 40%：情绪退潮，高位板谨慎（养家：高潮衰退期不碰高位板）")
if md["damian"] and md["damian"] > 0:
    signals.append(f"大面 {int(md['damian'])} 只（昨日涨停今日跌停）：亏钱效应出现")
if md["tiandi"] and md["tiandi"] > 0:
    signals.append(f"天地板 {int(md['tiandi'])} 只：接力情绪的剧烈预警")
if len(hist) >= 2 and hist["max_h"].iloc[-2] > md["max_h"]:
    signals.append(f"最高板 {int(hist['max_h'].iloc[-2])} → {int(md['max_h'])}：空间被压缩")
if md["temp"] >= 80:
    signals.append(f"情绪温度 {md['temp']:.0f}（高潮）：按心法此阶段不加仓，等分化")
st.markdown("**风险信号**（明日开盘前先看这些再定仓位）：" if signals else "**风险信号**：无明显退潮信号。")
for s in signals or ["-"]:
    st.markdown(f"- {s}")

