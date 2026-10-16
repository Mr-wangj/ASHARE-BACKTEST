"""看板页面：每日晋级明细。侧边栏的一进二明细、二进三明细、三进四明细三个页面共用本脚本，
板位由 app/dashboard.py 按当前页面传入（st.session_state["_page_level"]）。

每行 = 一只 T-1 首板（或二板、三板）股票在 T 日的表现。
- T-1日序号：T-1 首次涨停时间越早越靠前（越强）；
- T日序号：T 日竞价涨幅从高到低。
两个序号都基于当日全部首板候选计算，筛选只隐藏行、不改变序号。

查看方式：按 T 日（默认），或按首板日 T-1——后者包含最新交易日的首板（次日候选，key_preview=True），
收盘后自动更新完就能看到当天的首板；这些行的 T 日字段为空、显示“待开盘”。
页面顶部是日期与“只看 T 日竞价红”两个筛选（服务端）；明细表用自定义 HTML 组件（detail_table.html）渲染，
表头内置序号正序/倒序按钮、按涨停原因题材筛选，同题材同色，汇总指标与 CSV 导出都跟随表头筛选。
数据：scripts/run_backtest.py 生成的 data/features/daily_detail.parquet。
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ashare_bt.config import FEATURE_DIR, load_config
from ashare_bt.features.events import board_name, level_name
from ashare_bt.themes import assign_colors, load_theme_map, theme_table

DETAIL_PATH = FEATURE_DIR / "daily_detail.parquet"
TEMPLATE = Path(__file__).with_name("detail_table.html")
MAX_ROWS = 5000      # 区间太长时截断，避免页面过重
TABLE_MAX = 640      # 表格区域最大高度（像素），超出在表内滚动


@st.cache_data(show_spinner="加载每日明细…")
def load_detail(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(DETAIL_PATH)


def as_date(d: int) -> date:
    return datetime.strptime(str(int(d)), "%Y%m%d").date()


def as_int(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def fmt_day(d: int) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def num(v, scale: float = 1.0):
    """NaN/None → None（JSON null），其余转 float。"""
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    return float(v) * scale


def opt_int(v):
    return None if v is None or pd.isna(v) else int(v)


def short_day_fmt(days: pd.Series):
    """同一年内只显示月-日，跨年显示两位年份，节省列宽（CSV 里仍是完整日期）。"""
    same_year = (days // 10000).nunique() == 1
    return (lambda d: fmt_day(d)[5:]) if same_year else (lambda d: fmt_day(d)[2:])


def payload_rows(view: pd.DataFrame, th: pd.DataFrame, short) -> list[dict]:
    rows = []
    for idx, r in zip(view.index, view.itertuples(index=False)):
        t = th.loc[idx]
        pv = bool(r.key_preview)  # 次日候选：还没有 T 日
        rows.append({
            "preview": pv,
            "t1_rank": int(r.x_t1_rank), "t_rank": opt_int(r.x_t_rank), "bt": int(r.x_bt),
            "t1_date": fmt_day(r.key_t1_date), "t_date": "下一交易日" if pv else fmt_day(r.key_t_date),
            "t1_date_s": short(r.key_t1_date), "t_date_s": "待开盘" if pv else short(r.key_t_date),
            "code": r.key_code[4:], "name": r.key_name,
            "t1_auc_pct": num(r.x_t1_auc_pct), "t1_auc_amt": num(r.x_t1_auc_amt, 1e-4),
            "t_auc_pct": num(r.x_auc_pct), "t_auc_amt": num(r.x_t_auc_amt, 1e-4),
            "reason": "" if pd.isna(r.x_t1_reason) else str(r.x_t1_reason),
            "tags": t["tags"], "tag_cats": t["tag_cats"], "cats": t["cats"],
            "main": None if pd.isna(t["main"]) else str(t["main"]),
            "theme_n": opt_int(t["theme_n"]),
            "theme_rank_t1": opt_int(t["theme_rank_t1"]),   # 题材内T-1日序号
            "theme_rank_t": opt_int(t["theme_rank_t"]),     # 题材内T日序号
            "sealed": None if pv else bool(r.y_t_close_up), "touched": None if pv else bool(r.y_t_touch_up),
            "ret": num(r.y_t_ret_close),
            "first_time": None if pd.isna(r.x_t1_first_time) else str(r.x_t1_first_time),
        })
    return rows


lv = int(st.session_state.get("_page_level", 1))  # 板位由侧边栏页面决定（app/dashboard.py）：1 一进二、2 二进三、3 三进四
bd, lname = board_name(lv), level_name(lv)  # T-1 的板位（首板 / 二板 / 三板）与晋级名（一进二 / 二进三 / 三进四）
# 明细页左右留白收窄（默认 5rem），侧边栏展开时表格也不用横向滚动
st.html("<style>[data-testid='stMainBlockContainer'] { padding-left: 1.75rem; padding-right: 1.75rem; }</style>")
st.title(f"每日{lname}明细")
if not DETAIL_PATH.exists():
    st.error("未找到每日明细，请先运行：`uv run python scripts/download_limitup.py`，"
             "再运行 `uv run python scripts/run_backtest.py`")
    st.stop()

mtime = DETAIL_PATH.stat().st_mtime
df = load_detail(mtime)
for col, default in (("key_preview", False), ("key_level", 1)):  # 旧版明细表
    if col not in df.columns:
        df[col] = default
if not (df["key_level"] == lv).any():
    st.error(f"明细表里还没有{lname}数据，请先运行 `uv run python scripts/run_backtest.py` 重新生成。")
    st.stop()
universe = load_config()["universe"]
latest_t1 = int(df["key_t1_date"].max())

# ---------------------------------------------------------------- 查看方式 + 两个筛选（排序、题材筛选在表头）
MODES = {"t": f"按 T 日（看{lname}表现）", "t1": f"按{bd}日 T-1（含当日{bd} = 次日候选）"}
mode = st.segmented_control("查看方式", list(MODES), format_func=MODES.get, default="t", key="detail_mode",
                            required=True,
                            help=f"按 T 日：选的是买入日，看前一天的{bd}在这一天的表现；"
                                 f"按{bd}日：选的是涨停日，最新一个交易日收盘后就能看当天的{bd}（次日候选）")
date_col = "key_t_date" if mode == "t" else "key_t1_date"
base_all = df[~df["key_preview"]] if mode == "t" else df
days = sorted(int(d) for d in base_all[date_col].unique())  # 全部板位共用一组交易日：三板为空的日子也能选
first_day, last_day = as_date(days[0]), as_date(days[-1])
base = base_all[base_all["key_level"] == lv]
has_preview = bool(df.loc[df["key_level"] == lv, "key_preview"].any())

f1, f2, _ = st.columns([2.3, 1.8, 4], vertical_alignment="bottom")
picked = f1.date_input(("T日" if mode == "t" else f"{bd}日（T-1）") + "日期（可选一天或一段区间）",
                       value=(last_day, last_day), min_value=first_day, max_value=last_day,
                       format="YYYY-MM-DD", key=f"date_{mode}_{days[-1]}")  # 数据更新后默认跳到最新一天

if isinstance(picked, (tuple, list)):
    start, end = (picked[0], picked[-1]) if picked else (last_day, last_day)
else:
    start = end = picked
lo, hi = as_int(start), as_int(end)

day_df = base[base[date_col].between(lo, hi)]
only_preview = bool(len(day_df)) and bool(day_df["key_preview"].all())
red_only = f2.toggle("只看 T 日竞价红", value=False, disabled=only_preview, key=f"red_{mode}",
                     help="T 日竞价涨幅 > 0：9:25 集合竞价价格高于 T-1 收盘价"
                          + ("（次日候选还没有 T 日竞价）" if only_preview else ""))
view = day_df[day_df["x_auc_pct"] > 0] if red_only else day_df
n_days = day_df[date_col].nunique()
n_red = int((day_df["x_auc_pct"] > 0).sum())

rng = fmt_day(lo) + ("" if lo == hi else " ~ " + fmt_day(hi))
if day_df.empty:
    if any(lo <= d <= hi for d in days):  # 是交易日，只是这个板位没有候选（三板常有空的日子）
        st.info(f"{rng} 没有{bd}{'候选' if mode == 't' else ''}：{'T-1 当天' if mode == 't' else '当天'}"
                f"没有符合条件的{bd}，{lname}为空。可换一天，或在左侧切换到其他明细页。")
    else:
        st.warning(f"{rng} 没有数据（非交易日，或超出数据范围 {fmt_day(days[0])} ~ {fmt_day(days[-1])}）。")
    st.stop()
if only_preview:
    st.caption(f"{bd}日 {fmt_day(lo)}（最新交易日）：{bd} {len(day_df)} 只，是下一交易日的{lname}候选。"
               "T 日竞价、封板等数据在下一交易日收盘后（每日自动更新）补上。")
elif lo == hi and mode == "t":
    st.caption(f"T日 {fmt_day(lo)}，T-1日 {fmt_day(int(day_df['key_t1_date'].iloc[0]))}：{bd}候选 {len(day_df)} 只，"
               f"T 日竞价红 {n_red} 只")
elif lo == hi:
    st.caption(f"{bd}日 {fmt_day(lo)}，T日 {fmt_day(int(day_df['key_t_date'].iloc[0]))}：{bd} {len(day_df)} 只，"
               f"T 日竞价红 {n_red} 只")
else:
    n_pv = int(day_df["key_preview"].sum())
    st.caption(f"{rng}：{n_days} 个有{bd}的交易日，{bd}候选 {len(day_df)} 只"
               f"（日均 {len(day_df) / n_days:.1f}），T 日竞价红 {n_red} 只"
               + (f"；其中 {n_pv} 只为最新交易日{bd}（次日候选）" if n_pv else ""))
if mode == "t" and has_preview and hi >= int(base_all["key_t_date"].max()):
    st.caption(f"提示：{fmt_day(latest_t1)} 的{bd}（下一交易日的{lname}候选）在「按{bd}日」里查看。")
no_time = int(day_df["x_t1_first_sec"].isna().sum())
if no_time:
    bse = int((day_df["x_t1_first_sec"].isna() & (day_df["x_board"] == "bse")).sum())
    st.info(f"其中 {no_time} 只缺少 T-1 首次涨停时间（北交所 {bse} 只，其余为问财数据暂缺），T-1日序号排在当日最后；"
            "问财数据暂缺的日期可重跑 `scripts/download_limitup.py` 补齐。")
no_reason = int(day_df["x_t1_reason"].isna().sum())
if no_reason == len(day_df):
    st.info("当日涨停原因同花顺收盘后陆续发布，还没取到；每日 19:30 自动补跑一次，补到后即可按题材筛选和配色。"
            if only_preview else "所选日期的涨停原因还没下载（后台补数据进行中），暂时无法按题材筛选和配色。")

if len(view) > MAX_ROWS:
    st.warning(f"共 {len(view)} 行，只显示最近的 {MAX_ROWS} 行；请缩短日期区间。")
    view = view.sort_values("key_t_date", ascending=False).head(MAX_ROWS)

# ---------------------------------------------------------------- 明细表（自定义组件）
# 题材、两个题材内序号、配色都按所选日期的全部候选计算（不受“竞价红”影响），切换筛选时保持不变
tm = load_theme_map()
th = theme_table(day_df, tm)
main_counts = th["main"].dropna().value_counts().to_dict()
members: dict[str, set] = {}
for tags, cats in zip(th["tags"], th["tag_cats"]):
    for t, c in zip(tags, cats):
        members.setdefault(c, set()).add(t)
real_days = pd.concat([day_df["key_t1_date"], day_df.loc[~day_df["key_preview"], "key_t_date"]])
data = {
    "rows": payload_rows(view, th, short_day_fmt(real_days)),
    "colors": assign_colors(main_counts, tm),
    "members": {c: sorted(v, key=len) for c, v in members.items()},
    "group": "t_date" if mode == "t" else "t1_date",   # 多日时按哪一天分组排序
    "board": bd, "next_board": board_name(lv + 1),
    "table_max": TABLE_MAX,
    "filename": f"{lname}明细_{'T日' if mode == 't' else bd + '日'}_{lo}_{hi}{'_竞价红' if red_only else ''}",
}
html = TEMPLATE.read_text(encoding="utf-8").replace(
    "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
n = len(view)
n_view_days = view[date_col].nunique()
# 按日汇总：表头 + 最多 12 行，更多时在汇总表内滚动（见 detail_table.html），组件本身不滚动
days_block = 0 if n_days < 2 else 48 + 34 * (min(n_view_days, 12) + 1)
height = max(480, 70 + 64 + min(TABLE_MAX, 42 + 34 * max(n, 1)) + 56 + days_block)
components.html(html, height=height, scrolling=False)

st.caption(
    "说明：涨停原因来自同花顺问财「涨停原因类别」，按 `config/theme_groups.yaml` 的关键词归并成大类"
    "（如 AI漫剧、AI短剧、微短剧 → 短剧影视），可自行修改，保存后刷新即生效。"
    "「题材」= 主题材：按同花顺标签顺序取第一个题材类大类（跳过业绩、国资等事件类）；"
    "「题材内T-1日序号」= 同日同一主题材里按 T-1 首次涨停时间排第几，「题材内T日序号」= 按 T 日竞价涨幅排第几"
    "（如 1/13 = 13 只中的第 1）。"
    "表头「正序/倒序」按对应序号排序（多日时每天内排序），点其他列名也可排序，点「题材」列名按题材分组；"
    "「题材筛选」按主题材过滤，勾选「包含次要题材」后也匹配其他标签；同一题材同一种颜色，每行左侧色条是主题材。"
    "T-1日序号按同花顺“首次涨停时间”排序（开盘即涨停记 09:30:00）；T日序号按 T 日竞价涨幅"
    f"（9:25 集合竞价价 ÷ T-1 收盘 − 1）排序；两个序号都按当日全部{bd}候选计算，筛选只隐藏行、不改变序号。"
    "竞价成交额为 9:25 集合竞价成交额，涨停原因来自同花顺，数据经 thsdk（问财 + 日线）获取。"
    f"板位：首板 = 当日涨停且前 {universe['first_board_lookback']} 个交易日未涨停；二板 = 首板次日再涨停"
    "（连续 2 个交易日涨停，即前一天一进二的封板股），三板同理；停牌不打断连板，与行情软件口径一致。"
    f"股票池与回测一致：剔除 ST、上市不足 {universe['min_list_days']} 个交易日的次新股。"
)
