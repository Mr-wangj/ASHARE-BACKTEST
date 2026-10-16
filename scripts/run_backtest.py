"""一进二弱转强 MVP 回测：uv run python scripts/run_backtest.py"""
from __future__ import annotations

import logging
import sys
from datetime import datetime

import pandas as pd

from ashare_bt.analysis import AUC_LABELS, BOARD_NAMES, auc_band, quarterly_table, segment_masks, tercile
from ashare_bt.backtest.metrics import by_group, compare_vs_rest, pnl_structure, summarize
from ashare_bt.backtest.portfolio import curve_stats, equity_curve
from ashare_bt.config import FEATURE_DIR, RAW_DIR, REPORT_DIR, ensure_dirs, load_config
from ashare_bt.data.wencai_limitup import load_limitup
from ashare_bt.features.events import DETAIL_LEVELS, RETURN_RULES, board_name, build_events, compute_returns, level_name
from ashare_bt.features.ladder import KIND_DK, KIND_QB, KIND_ZB, KIND_ZD, build_down_ladder, build_ladder
from ashare_bt.features.limit import BT_NAMES
from ashare_bt.features.mood import market_mood
from ashare_bt.features.panel import build_panel, eligible_mask
from ashare_bt.features.sentiment import daily_sentiment
from ashare_bt.features.strength import STRENGTH_MODES
from ashare_bt.report import df_to_md, plot_bars, plot_curves
from ashare_bt.strategy.weak_to_strong import VARIANT_NAMES, VARIANTS, select

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("run")


def load_raw():
    sec = pd.read_parquet(RAW_DIR / "securities.parquet")
    daily = pd.read_parquet(RAW_DIR / "daily_all.parquet")
    fs = pd.read_parquet(RAW_DIR / "float_shares.parquet")
    idx = pd.read_parquet(RAW_DIR / "index_daily.parquet")
    calendar = sorted(idx.loc[idx["index"] == "sh", "date"].astype("int64").unique().tolist())
    return sec, daily, fs, idx, calendar


def validate_limit(panel: pd.DataFrame, elig: pd.Series, start: int) -> tuple[dict, pd.DataFrame]:
    w = panel[elig & (panel["date"] >= start)]
    s, t = w["is_up"], w["ths_up"].astype(bool)
    both, only_s, only_t = int((s & t).sum()), int((s & ~t).sum()), int((~s & t).sum())
    pct_norm = (w["close"] / w["prev_close"] - 1) / w["limit_pct"]
    mism = w[s ^ t][["full_code", "name", "board", "date", "open", "high", "low", "close",
                     "prev_close", "up_limit", "is_up", "ths_up"]].copy()
    mism["标准化涨幅"] = pct_norm[s ^ t]
    st_like = ((pct_norm[~s & t] - 0.5).abs() < 0.08).mean() if only_t else float("nan")
    only_t_non_st = int((~s & t & ((pct_norm - 0.5).abs() >= 0.08)).sum())
    return {
        "自算与同花顺均为涨停": both,
        "仅自算涨停": only_s,
        "仅同花顺涨停": only_t,
        "一致率": both / max(both + only_s + only_t, 1),
        "一致率(剔除历史ST日)": both / max(both + only_s + only_t_non_st, 1),
        "仅同花顺·标准化涨幅中位数": float(pct_norm[~s & t].median()) if only_t else float("nan"),
        "仅同花顺·疑似5%涨停占比": float(st_like),
    }, mism


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    u, sp, costs, bt = cfg["universe"], cfg["strategy"], cfg["costs"], cfg["backtest"]

    sec, daily, fs, index_daily, calendar = load_raw()
    log.info("原始日线 %d 行，%d 只；交易日 %d（%s~%s）", len(daily), daily["full_code"].nunique(),
             len(calendar), calendar[0], calendar[-1])

    panel = build_panel(daily, sec, fs, calendar, first_board_lookback=u["first_board_lookback"])
    elig = eligible_mask(panel, exclude_st=u["exclude_st"], min_list_days=u["min_list_days"])
    check, mism = validate_limit(panel, elig, bt["start"])
    mism.to_csv(REPORT_DIR / "limit_mismatch.csv", index=False, encoding="utf-8-sig")
    log.info("涨停校验：%s", check)

    sent = daily_sentiment(panel, elig)
    limitup = load_limitup()
    if limitup is None and sp["strength_mode"] == "time":
        raise SystemExit("strength_mode=time 需要涨停明细，请先运行：uv run python scripts/download_limitup.py")
    ev = build_events(panel, sent, calendar, cfg, limitup=limitup)

    # 每日晋级明细：一进二、二进三、三进四（含最后一个交易日 + 当日涨停作为次日候选，供看板使用；不参与回测）
    detail = pd.concat([build_events(panel, sent, calendar, cfg, limitup=limitup, include_last_day=True,
                                     include_preview=True, level=lv) for lv in DETAIL_LEVELS], ignore_index=True)
    detail.to_parquet(FEATURE_DIR / "daily_detail.parquet", index=False)
    hist = detail[~detail["key_preview"] & (detail["key_level"] == 1)]
    lu_cov = {
        "首次涨停时间": float(hist["x_t1_first_sec"].notna().mean()),
        "涨停原因": float(hist["x_t1_reason"].notna().mean()),
        "T-1竞价成交额": float(hist["x_t1_auc_amt"].notna().mean()),
        "T日竞价成交额": float(hist["x_t_auc_amt"].notna().mean()),
    }
    per_level = []
    for lv in DETAIL_LEVELS:
        d = detail[detail["key_level"] == lv]
        h, p = d[~d["key_preview"]], d[d["key_preview"]]
        per_level.append(f"{level_name(lv)} {len(h)} 行（日均 {len(h) / max(h['key_t_date'].nunique(), 1):.1f}，"
                         f"封板率 {h['y_t_close_up'].mean():.1%}；{calendar[-1]} {board_name(lv)} {len(p)} 只，"
                         f"有涨停原因 {int(p['x_t1_reason'].notna().sum())}）")
    log.info("每日明细（T 日至 %s）：%s；问财明细覆盖率（首板）%s", hist["key_t_date"].max(), "；".join(per_level),
             {k: f"{v:.1%}" for k, v in lu_cov.items()})

    # 涨停天梯（看板「涨停天梯」页）与每日情绪指标；从问财明细的首日开始
    lad_start = int(limitup["t1_date"].min()) if limitup is not None and not limitup.empty else bt["start"]
    ladder = build_ladder(panel, calendar, cfg, limitup=limitup, start=lad_start)
    ladder.to_parquet(FEATURE_DIR / "limitup_ladder.parquet", index=False)
    sent.rename_axis("date").reset_index().to_parquet(FEATURE_DIR / "sentiment.parquet", index=False)
    last = ladder[ladder["date"] == calendar[-1]]
    log.info("涨停天梯 %d 行（%s~%s）；%s 涨停 %d、炸板 %d、断板 %d，最高 %d 板", len(ladder), lad_start, calendar[-1],
             calendar[-1], int(last["is_up"].sum()), int((last["kind"] == KIND_ZB).sum()),
             int((last["kind"] == KIND_DK).sum()), int(last["streak"].max()) if len(last) else 0)
    down = build_down_ladder(panel, cfg, start=lad_start)
    down.to_parquet(FEATURE_DIR / "limitdown_ladder.parquet", index=False)
    dl = down[down["date"] == calendar[-1]]
    log.info("跌停天梯 %d 行；%s 跌停 %d、翘板 %d、止跌 %d，最高连跌 %d 天", len(down), calendar[-1],
             int(dl["is_down"].sum()), int((dl["kind"] == KIND_QB).sum()), int((dl["kind"] == KIND_ZD).sum()),
             int(dl["down_streak"].max()) if len(dl) else 0)

    # 情绪监测（看板「情绪监测」页）：每个交易日的大盘、涨跌停、高度、晋级、溢价、亏钱效应指标与情绪温度
    mood = market_mood(panel, elig, index_daily, limitup=limitup)
    mood.to_parquet(FEATURE_DIR / "market_mood.parquet", index=False)
    chk = mood.set_index("date")[["zt", "dt", "zb_rate", "max_h"]].join(
        sent[["zt_count", "dt_count", "zb_rate", "max_height"]].add_prefix("s_"))
    bad = ((chk["zt"] != chk["s_zt_count"]) | (chk["dt"] != chk["s_dt_count"]) | (chk["max_h"] != chk["s_max_height"])
           | ((chk["zb_rate"] - chk["s_zb_rate"]).abs() > 1e-12)).sum()
    if bad:
        log.warning("情绪监测与每日情绪指标不一致的交易日 %d 个，请检查", bad)
    m = mood.iloc[-1]
    log.info("情绪监测 %d 个交易日（%s~%s）；%s 情绪温度 %s（%s）：两市成交 %.2f 万亿，涨停 %d、跌停 %d、炸板率 %.0f%%、"
             "最高 %d 板、最高连跌 %d 天", len(mood), mood["date"].iloc[0], mood["date"].iloc[-1], m["date"],
             "-" if pd.isna(m["temp"]) else f"{m['temp']:.0f}", m["temp_level"], m["turnover"] / 1e12, m["zt"], m["dt"],
             (m["zb_rate"] or 0) * 100, m["max_h"], m["max_dh"])
    if u.get("float_mv_min") is not None:
        ev = ev[ev["x_t1_float_mv"] >= u["float_mv_min"]]
    if u.get("float_mv_max") is not None:
        ev = ev[ev["x_t1_float_mv"] <= u["float_mv_max"]]
    ev = ev.reset_index(drop=True)
    ev.to_parquet(FEATURE_DIR / "events.parquet", index=False)
    rets = compute_returns(ev, costs)
    log.info("事件宽表 %d 行，T 日 %s~%s", len(ev), ev["key_t_date"].min(), ev["key_t_date"].max())

    t_dates = [d for d in calendar if ev["key_t_date"].min() <= d <= ev["key_t_date"].max()]
    segs = segment_masks(ev, bt["in_sample_end"])
    top_n, min_pool = sp["top_n"], u["min_pool"]
    sel_all = {v: select(ev, v, sp, min_pool) for v in VARIANTS}
    sel_top = {v: select(ev, v, sp, min_pool, top_n=top_n) for v in VARIANTS}

    # ---------- 核心结果
    summary_tables = {}
    for seg, m in segs.items():
        rows = []
        for v in VARIANTS:
            idx = sel_all[v][m.loc[sel_all[v]].to_numpy()]
            rows.append({"口径": VARIANT_NAMES[v], **summarize(ev, rets, idx)})
        summary_tables[seg] = pd.DataFrame(rows)

    # ---------- 显著性：信号组 vs 同池其余样本
    cmp_rows = []
    for seg, m in segs.items():
        for v in ["A", "B", "C"]:
            idx = sel_all[v][m.loc[sel_all[v]].to_numpy()]
            for pool_v in (["base1", "base3"] if v != "C" else ["base1"]):
                pool = sel_all[pool_v][m.loc[sel_all[pool_v]].to_numpy()]
                cmp_rows.append({"区间": seg, "口径": VARIANT_NAMES[v], "对照池": VARIANT_NAMES[pool_v],
                                 **compare_vs_rest(ev, idx, pool)})
    cmp_df = pd.DataFrame(cmp_rows)

    # ---------- 盈亏结构：封板 vs 未封板
    pnl = {v: pnl_structure(ev, rets, sel_all[v]) for v in ["base1", "base3", "A", "B", "C"]}

    # ---------- 分层
    auc_label = auc_band(ev)
    weak_mask = ev["x_P_str"] >= sp["p_str_min"]
    strong_mask = ev["x_P_str"] < sp["p_str_min"]  # 显式写出：缺少强度（P_str 为空）的不计入任一组
    tradable = sel_all["base1"]
    auc_weak = by_group(ev, rets, tradable[weak_mask.loc[tradable].to_numpy()], auc_label)
    auc_strong = by_group(ev, rets, tradable[strong_mask.loc[tradable].to_numpy()], auc_label)

    main_v = "B"
    main_idx = sel_all[main_v]
    layer = {
        "板块": ev["x_board"].map(BOARD_NAMES),
        "T-1首板晋级率(三分位)": tercile(ev["x_s_promo_rate_t1"]),
        "T日昨日涨停竞价均值(三分位)": tercile(ev["x_s_yzt_auc_T"]),
        "T-1涨停家数(三分位)": tercile(ev["x_s_zt_count_t1"]),
        "T-1流通市值(三分位)": tercile(ev["x_t1_float_mv"], ("小", "中", "大")),
        "月份": (ev["key_t_date"] // 100).astype(str),
    }
    layer_tables = {name: by_group(ev, rets, main_idx, g) for name, g in layer.items()}
    # 情绪分层同样对基准1做一遍，判断情绪过滤是否普适
    layer_base1 = {name: by_group(ev, rets, sel_all["base1"], layer[name])
                   for name in ["T-1首板晋级率(三分位)", "T日昨日涨停竞价均值(三分位)"]}
    # 板型：弱板几乎都是实体板，故对全部首板（基准1）统计 T-1 板型
    bt_label = pd.Categorical(ev["x_bt"].map(BT_NAMES), categories=["一字板", "T字板", "实体板"], ordered=True)
    bt_base1 = by_group(ev, rets, sel_all["base1"], pd.Series(bt_label, index=ev.index))

    # ---------- 组合（每日最多 top_n 只；基准为全部等权）
    port_rows, curves_r1, curves_r3 = [], {}, {}
    for v in ["base1", "base3", "A", "B", "C"]:
        idx = sel_top[v]
        for rule in ["r1", "r3"]:
            c = equity_curve(ev, rets, idx, rule, t_dates)
            st = curve_stats(c)
            port_rows.append({"口径": VARIANT_NAMES[v] + ("" if v.startswith("base") else f"·前{top_n}"),
                              "卖出规则": RETURN_RULES[rule], **st})
            (curves_r1 if rule == "r1" else curves_r3)[VARIANT_NAMES[v]] = c
    port_df = pd.DataFrame(port_rows)

    # ---------- 按季度（信号口径每日前 top_n 只）
    q_df = quarterly_table(ev, rets, {v: sel_top[v] for v in ["A", "B", "C"]}, rule="r1")

    # ---------- 滑点敏感性（主口径，全部信号）
    slip_rows = []
    for slip in [0.0, 0.003, 0.005, 0.01]:
        r_s = compute_returns(ev, {**costs, "slippage": slip})
        s = summarize(ev, r_s, main_idx)
        c = curve_stats(equity_curve(ev, r_s, sel_top[main_v], "r3", t_dates))
        slip_rows.append({"滑点(单边)": f"{slip:.1%}", "r1均值": s["r1均值"], "r1胜率": s["r1胜率"],
                          "r3均值": s["r3均值"], "r3胜率": s["r3胜率"], "累计收益": c["累计收益"],
                          "最大回撤": c["最大回撤"]})
    slip_df = pd.DataFrame(slip_rows)

    # ---------- 输出明细
    for v in ["A", "B", "C"]:
        out = ev.loc[sel_all[v]].join(rets)
        out["top_n入选"] = out.index.isin(sel_top[v])
        out.to_csv(REPORT_DIR / f"signals_{v}.csv", index=False, encoding="utf-8-sig")

    # ---------- 图
    plot_curves(curves_r1, f"组合净值（T+1开盘卖，信号口径每日前{top_n}只，含费用滑点）", REPORT_DIR / "equity_r1.png")
    plot_curves(curves_r3, f"组合净值（封板者T+1收盘卖/否则开盘卖，每日前{top_n}只）", REPORT_DIR / "equity_r3.png")
    aw = auc_weak[["分组", "封板率"]].rename(columns={"封板率": "弱板"})
    as_ = auc_strong[["分组", "封板率"]].rename(columns={"封板率": "强板"})
    band = aw.merge(as_, on="分组", how="outer")
    band["分组"] = pd.Categorical(band["分组"].astype(str), categories=AUC_LABELS, ordered=True)
    plot_bars(band.sort_values("分组"), "分组", ["弱板", "强板"],
              "T日竞价标准化涨幅 vs 当日封板率（剔除开盘即涨停）", REPORT_DIR / "auc_band.png")
    mon = layer_tables["月份"].copy()
    mon_b1 = by_group(ev, rets, sel_all["base1"], layer["月份"])[["分组", "封板率"]].rename(columns={"封板率": "基准1"})
    mon = mon[["分组", "封板率"]].rename(columns={"封板率": VARIANT_NAMES[main_v]}).merge(mon_b1, on="分组")
    plot_bars(mon, "分组", [VARIANT_NAMES[main_v], "基准1"], "月度封板率", REPORT_DIR / "monthly.png")

    # ---------- 报告
    def rate(v, seg="全样本"):
        tb = summary_tables[seg]
        return float(tb.loc[tb["口径"] == VARIANT_NAMES[v], "封板率"].iloc[0])

    def stat(v, col, seg="全样本"):
        tb = summary_tables[seg]
        return float(tb.loc[tb["口径"] == VARIANT_NAMES[v], col].iloc[0])

    best = max(["A", "B", "C"], key=lambda v: rate(v, "样本外"))
    pnl_best, be_best = pnl[best]
    hit_r1 = float(pnl_best.loc[pnl_best["分组"] == "封板", "r1均值"].iloc[0])
    miss_r1 = float(pnl_best.loc[pnl_best["分组"] == "未封板", "r1均值"].iloc[0])
    port_best = port_df[(port_df["口径"].str.startswith(VARIANT_NAMES[best])) & (port_df["卖出规则"] == RETURN_RULES["r1"])]
    pos_q = []
    for _, row in q_df.iterrows():
        hits = [v for v in ["A", "B", "C"] if row[f"{v}·样本数"] >= 20 and row[f"{v}·r1均值"] > 0]
        if hits:
            pos_q.append(f"{row['季度']}（{'/'.join(hits)}）")
    pos_q_text = "、".join(pos_q) if pos_q else "无"
    conclusion = [
        "## 0. 核心结论",
        "",
        f"1. **弱转强显著提高封板率，且样本外成立。** 全部首板次日竞价买入（基准1）的封板率为 "
        f"{rate('base1'):.1%}；三个弱转强口径为 {rate('A'):.1%}（A）、{rate('B'):.1%}（B）、{rate('C'):.1%}（C）。"
        f"样本外分别为 {rate('A', '样本外'):.1%}、{rate('B', '样本外'):.1%}、{rate('C', '样本外'):.1%}，基准1 为 "
        f"{rate('base1', '样本外'):.1%}。",
        f"2. **提升不只是“高开”带来的。** 与“弱板 + 同样竞价区间”（基准3，{rate('base3'):.1%}）相比，"
        f"口径A 仍高出约 {(rate('A') - rate('base3')) * 100:.0f} 个百分点（见第 3 节检验），说明“相对强板组反超”本身有信息量。",
        f"3. **但扣费后单笔期望为负。** 以样本外最好的口径{best}为例：封板者 T+1 开盘卖平均 {hit_r1:+.2%}，"
        f"未封板者平均 {miss_r1:+.2%}；封板率要达到 **{be_best:.1%}** 才能盈亏平衡，实际只有 {rate(best):.1%}。"
        f"r1 均值 {stat(best, 'r1均值'):+.2%}，每日前 {top_n} 只的组合累计 {float(port_best['累计收益'].iloc[0]):+.1%}。",
        "4. **竞价越高，封板率越高，但收益没有跟着提高。** 高开吃掉了涨停空间，未封板时回落更深（见 4.1）。"
        "因此只按“封板胜率”选参数会误导，必须同时看 T+1 卖出后的净收益。",
        f"5. **滑点很敏感。** 单边滑点从 0.3% 降到 0，口径B 的 r1 均值从 {slip_df.loc[1, 'r1均值']:+.2%} "
        f"变为 {slip_df.loc[0, 'r1均值']:+.2%}，仍为负。",
        f"6. **阶段性明显。** 按季度看（样本 ≥ 20），每日前 {top_n} 只的单笔均值为正的季度只有："
        f"{pos_q_text}；其余季度均为负（见第 5 节季度表）。说明需要情绪择时，"
        "但本版的简单情绪三分位还没找到稳定的开关（见 4.2）。",
        "",
        ("**下一步最值得做的**：把 9:25 竞价成交额（量能）加入信号过滤，强度里补充炸板次数、封单额，"
         "并把卖出从“T+1 开盘一刀切”改为“封板者冲高卖、未封板者竞价止损”的分钟级规则。"
         if sp["strength_mode"] == "time" else
         "**下一步最值得做的**：用首次涨停时间替换日线代理强度（strength_mode: time），加入竞价成交额，"
         "并把卖出从“T+1 开盘一刀切”改为“封板者冲高卖、未封板者竞价止损”的分钟级规则。"),
        "",
    ]
    pool_days = ev.groupby("key_t_date")["x_pool_n"].first()
    board_cnt = sec["board"].map(BOARD_NAMES).value_counts().to_dict()
    lines = [
        "# 一进二·弱转强 回测报告（MVP）",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- 数据源：thsdk（同花顺），不复权日线 + 交易所前收；股票 {len(sec)} 只（{board_cnt}）",
        f"- 回测区间（T 日）：{ev['key_t_date'].min()} ~ {ev['key_t_date'].max()}，共 {len(t_dates)} 个交易日；"
        f"样本内 ≤ {bt['in_sample_end']}，之后为样本外",
        f"- 股票池：剔除当前 ST、T-1 上市不足 {u['min_list_days']} 个交易日；首板 = T-1 涨停且前 "
        f"{u['first_board_lookback']} 个交易日无涨停；当日首板 < {min_pool} 只不交易",
        f"- 成本：佣金 {costs['commission']:.4%}×2、印花税 {costs['stamp_duty']:.3%}（卖）、过户费 "
        f"{costs['transfer_fee']:.4%}×2、滑点 {costs['slippage']:.1%}×2",
        f"- 参数：弱板 P_str ≥ {sp['p_str_min']}；竞价标准化涨幅 ∈ [{sp['auc_norm_lo']}, {sp['auc_norm_hi']}]；"
        f"口径A P_auc ≤ {sp['p_auc_max']}；口径B 强板组 P_str ≤ {sp['strong_group_max']}；口径C Δ ≥ {sp['delta_min']}；"
        f"强度：{STRENGTH_MODES.get(sp['strength_mode'], sp['strength_mode'])}",
        "",
        *conclusion,
        "## 1. 数据校验",
        "",
        f"- 自算涨停（非ST规则）与同花顺涨停事件：剔除历史 ST 日后一致率 **{check['一致率(剔除历史ST日)']:.2%}**"
        f"（均涨停 {check['自算与同花顺均为涨停']}，仅自算 {check['仅自算涨停']}，仅同花顺 {check['仅同花顺涨停']}；"
        f"含 ST 日的原始一致率 {check['一致率']:.2%}）",
        f"- 仅同花顺涨停的样本中，{check['仅同花顺·疑似5%涨停占比']:.0%} 为 5% 涨停（当前已摘帽股票的历史 ST 日，按设计排除）；"
        "仅自算涨停多为收盘价等于涨停价但同花顺未计入（口径差异或事件缺失）；明细见 `reports/limit_mismatch.csv`",
        "- 北交所涨跌停价规则已按事件数据校准：涨停价向下截断、跌停价向上取整（四舍五入会导致约一半北交所涨停被漏判）",
        f"- 首板事件 {len(ev)} 个；日均首板候选 {pool_days.mean():.1f} 只（中位数 {pool_days.median():.0f}）；"
        f"首板 < {min_pool} 只的天数 {(pool_days < min_pool).sum()}",
        f"- T 日开盘即涨停（不可买入，已剔除）：{int(ev['x_open_limit'].sum())} 个，占 {ev['x_open_limit'].mean():.1%}",
        "- 问财涨停明细（首次涨停时间、涨停原因、9:25 竞价成交额）对首板候选的覆盖率："
        + "，".join(f"{k} {v:.1%}" for k, v in lu_cov.items())
        + "；缺少首次涨停时间的候选在 T-1日序号中排在当日最后",
        "",
        "## 2. 核心结果（全部信号，不限每日只数）",
        "",
        "封板率 = T 日收盘封住涨停的比例（用户定义的胜率）；r1~r4 为扣除费用滑点后的单笔净收益：",
        "",
    ]
    for rule, name in RETURN_RULES.items():
        lines.append(f"- {rule}：{name}")
    cols = ["口径", "样本数", "交易天数", "封板率", "CI下", "CI上", "触板率", "炸板率", "T日收盘浮盈",
            "r1均值", "r1胜率", "r1盈亏比", "r3均值", "r3胜率", "r4均值"]
    for seg, tb in summary_tables.items():
        lines += ["", f"### {seg}", "", df_to_md(tb, cols)]
    lines += ["", "### 盈亏结构：封板 vs 未封板（全样本）", "",
              "盈亏平衡封板率 = 在 r1 规则下，封板率需要达到多少才能让单笔均值转正。", ""]
    for v, (tb, be) in pnl.items():
        lines += [f"**{VARIANT_NAMES[v]}**：实际封板率 {ev.loc[sel_all[v], 'y_t_close_up'].mean():.1%}，"
                  f"盈亏平衡封板率 **{be:.1%}**", "",
                  df_to_md(tb, ["分组", "样本数", "占比", "T日收盘浮盈", "T+1开盘溢价", "r1均值", "r1胜率", "r2均值", "r3均值"]), ""]
    lines += [
        "", "## 3. 弱转强是否带来增量（信号组 vs 同池其余样本，两比例 z 检验）", "",
        "对照基准3可以区分：提升来自“相对强板组反超”，还是仅来自“绝对高开”。", "",
        df_to_md(cmp_df, ["区间", "口径", "对照池", "信号封板率", "池内其余封板率", "提升(百分点)", "z", "p值"]),
        "", "## 4. 分层统计", "",
        "### 4.1 T 日竞价标准化涨幅 vs 封板率（全部可买入首板，按 T-1 强弱拆分）", "",
        "![auc](auc_band.png)", "", "**弱板（P_str ≥ 阈值）**", "",
        df_to_md(auc_weak), "", "**强板（P_str < 阈值）**", "", df_to_md(auc_strong), "",
        f"### 4.2 主口径（{VARIANT_NAMES[main_v]}）分层", "",
    ]
    for name, tb in layer_tables.items():
        if name == "月份":
            continue
        lines += [f"**{name}**", "", df_to_md(tb), ""]
    lines += ["**T-1 板型·基准1（全部首板次日；弱板几乎全为实体板）**", "", df_to_md(bt_base1), ""]
    lines += ["**情绪分层·基准1（检验情绪过滤是否普适）**", ""]
    for name, tb in layer_base1.items():
        lines += [f"*{name}*", "", df_to_md(tb), ""]
    lines += ["**月度**", "", "![monthly](monthly.png)", "", df_to_md(layer_tables["月份"]), ""]
    lines += [
        f"## 5. 组合回测（信号口径每日按 Δ 取前 {top_n} 只；基准为全部等权；资金分两份隔日轮换）", "",
        df_to_md(port_df, ["口径", "卖出规则", "累计收益", "年化收益", "最大回撤", "夏普", "有信号天数", "总天数"]),
        "", "![r1](equity_r1.png)", "", "![r3](equity_r3.png)", "",
        f"### 按季度（信号口径每日前 {top_n} 只，r1 = T+1 开盘卖净收益）", "",
        df_to_md(q_df), "",
        f"### 滑点敏感性（{VARIANT_NAMES[main_v]}；组合为 r3 规则、每日前 {top_n} 只）", "",
        df_to_md(slip_df), "",
        "## 6. MVP 局限（影响结论可信度，需在完整版解决）", "",
        ("1. **强度只看首次涨停时间**：T-1 按同花顺首次涨停时间排序（越早越强），未考虑炸板次数、封单额；"
         "同一时间（如开盘即涨停）按板型、T-1 竞价涨幅排序。" if sp["strength_mode"] == "time" else
         "1. **强度排序为日线代理**：“是否炸板、首次封板时间”用“开盘涨幅 + 换手率 + 盘中回撤”代理；"
         "板型（一字/T字/实体）是精确的。"),
        "2. **竞价量能未进入信号**：竞价成交额（9:25）已取到并在每日明细页展示，但口径 A/B/C 尚未使用量能过滤。",
        "3. **竞价信息时差**：开盘价同时做信号和成交价，实盘需 9:24 按虚拟价下单或 9:30 后追买，已用滑点近似。",
        "4. **流通股本为当前快照**：历史流通市值、换手率为近似值（解禁、增发会带来偏差）。",
        "5. **幸存者偏差**：股票列表为当前上市股票，不含回测期内已退市股票。",
        "6. **ST**：剔除当前 ST；历史 ST 日（5% 涨跌幅）因达不到 10% 涨停价被天然排除。",
        "",
        "明细：`reports/signals_A.csv`、`signals_B.csv`、`signals_C.csv`；事件宽表：`data/features/events.parquet`；"
        "每日晋级明细（一进二、二进三、三进四）：`data/features/daily_detail.parquet`。",
    ]
    (REPORT_DIR / "mvp_report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("报告已生成：%s", REPORT_DIR / "mvp_report.md")

    # 控制台摘要
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = ["口径", "样本数", "交易天数", "封板率", "CI下", "CI上", "触板率", "r1均值", "r1胜率", "r3均值"]
    for seg, tb in summary_tables.items():
        print(f"\n== {seg}")
        print(tb[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n== 显著性")
    print(cmp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n== 组合")
    print(port_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
