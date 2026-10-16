"""市场情绪日度指标（看板「情绪监测」页；scripts/run_backtest.py 生成 data/features/market_mood.parquet）。

两种股票范围：
- 大盘（成交额、涨跌家数、涨跌中位数、涨超 / 跌超 5%）：当日有成交的全部 A 股（沪深京，含 ST、次新）；两市成交额 = 沪深；
- 涨跌停相关：与天梯、明细页相同的股票池（剔除 ST、上市不足 N 个交易日的次新股），涨跌停价按非 ST 规则，
  连板 / 连跌天数停牌不打断。
“昨日 X 溢价”= 昨日（相邻交易日）为 X 的股票今日收盘涨跌幅的等权平均（今日停牌的不计）。

情绪温度：TEMP_COMPONENTS 里 9 个指标各自在近 window 个交易日里的分位（跌停家数、大面取反，越多越冷），
等权平均后 ×100；至少 7 个指标有值才计算。< 20 冰点、20~40 偏冷、40~60 中性、60~80 偏热、≥ 80 高潮。
分位是相对近一年而言的：同样 60 家涨停，在冷清的年份可能算偏热，在火热的年份可能只算中性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 0.0049          # 价格比较容差（不到 1 分钱算平盘）
EARLY_SEC = 10 * 3600  # 10:00 前涨停
TEMP_COMPONENTS = {    # 指标 → 方向（+1 越大越热，-1 越大越冷）
    "zt": 1, "lb": 1, "max_h": 1, "seal_rate": 1, "promo_fb": 1, "prem_zt": 1, "rise_ratio": 1, "dt": -1, "damian": -1,
}
TEMP_NAMES = {"zt": "涨停家数", "lb": "连板家数", "max_h": "涨停高度", "seal_rate": "封板率", "promo_fb": "首板晋级率",
              "prem_zt": "昨日涨停溢价", "rise_ratio": "上涨占比", "dt": "跌停家数（反向）", "damian": "大面（反向）"}
LEVELS = ((20, "冰点"), (40, "偏冷"), (60, "中性"), (80, "偏热"), (float("inf"), "高潮"))
LEVEL_NAMES = [name for _, name in LEVELS]


def temp_level(t) -> str | None:
    if t is None or not np.isfinite(t):
        return None
    return next(name for hi, name in LEVELS if t < hi)


def rolling_pct(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """当日值在近 window 个值（含当日、忽略空值）里的分位，(名次 - 0.5) / 个数，落在 (0, 1)。"""
    r = s.rolling(window, min_periods=min_periods)
    return (r.rank() - 0.5) / r.count()


def _names(df: pd.DataFrame, col: str, top: int) -> pd.Series:
    """每日 col 最高的股票名称（最多 top 个）。"""
    if df.empty:
        return pd.Series(dtype="object")
    mx = df.groupby("date")[col].transform("max")
    tops = df[df[col] == mx]
    return tops.groupby("date")["name"].agg(
        lambda s: "、".join(s.astype(str).head(top)) + (f" 等 {len(s)} 只" if len(s) > top else ""))


def market_mood(panel: pd.DataFrame, elig: pd.Series, index_daily: pd.DataFrame | None = None, *,
                limitup: pd.DataFrame | None = None, window: int = 250, min_periods: int = 60, lookback: int = 20,
                top: int = 4) -> pd.DataFrame:
    """panel：features.panel.build_panel 的结果（按 股票、日期 排序）；elig：涨跌停指标的股票池；
    index_daily：指数日线（index ∈ sh / sz / gem）；limitup：问财涨停明细（10 点前涨停用）。返回每个交易日一行。"""
    p = panel
    d = p["date"].to_numpy()
    traded = p["volume"].fillna(0).to_numpy() > 0
    diff = (p["close"] - p["prev_close"]).to_numpy(float)
    pct = (p["close"] / p["prev_close"] - 1).to_numpy(float)
    opct = (p["open"] / p["prev_close"] - 1).to_numpy(float)
    mkt = p["full_code"].str[:4].to_numpy()
    e = elig.to_numpy(bool)
    up, dn = p["is_up"].to_numpy(bool), p["is_down"].to_numpy(bool)
    tu, td = p["touch_up"].to_numpy(bool), p["touch_down"].to_numpy(bool)
    streak, dstreak = p["streak"].to_numpy(), p["down_streak"].to_numpy()
    adj = (p["prv_cal_i"] == p["cal_i"] - 1).to_numpy()
    pu = adj & p["prv_is_up"].fillna(False).astype(bool).to_numpy()
    pfb = adj & p["prv_is_first_board"].fillna(False).astype(bool).to_numpy()
    pdn = adj & p["prv_is_down"].fillna(False).astype(bool).to_numpy()
    pstreak = p["prv_streak"].fillna(0).to_numpy()
    same = p["full_code"].eq(p["full_code"].shift(1)).to_numpy()
    pzb = adj & same & p["touch_up"].shift(1, fill_value=False).astype(bool).to_numpy() & ~pu   # 昨日炸板
    max20 = (p.groupby("full_code", sort=False)["streak"].rolling(lookback, min_periods=1).max()
             .reset_index(level=0, drop=True).reindex(p.index).to_numpy())

    flags = {
        "n_stock": traded, "n_rise": traded & (diff > EPS), "n_fall": traded & (diff < -EPS),
        "n_up5": traded & (pct >= 0.05), "n_dn5": traded & (pct <= -0.05),
        "zt": e & up, "dt": e & dn, "touch": e & tu, "fb": e & p["is_first_board"].to_numpy(bool),
        "lb": e & up & (streak >= 2), "b1": e & up & (streak == 1), "b2": e & up & (streak == 2),
        "b3": e & up & (streak >= 3), "yizi": e & up & (p["board_type"].to_numpy() == 3),
        "dt_lb": e & dn & (dstreak >= 2), "damian": e & dn & pu, "tiandi": e & dn & tu, "ditian": e & up & td,
        "high_fall": e & dn & (max20 >= 3),
        "_promo_fb": e & pfb & up, "_promo_lb": e & pu & (pstreak >= 2) & up, "_promo_all": e & pu & up,
    }
    out = pd.DataFrame(flags).groupby(d).sum()
    amt = p["amount"].fillna(0).to_numpy(float)
    sums = pd.DataFrame({"turnover": np.where(np.isin(mkt, ["USHA", "USZA"]) & traded, amt, 0.0),
                         "turnover_bj": np.where((mkt == "USTM") & traded, amt, 0.0),
                         "zt_amount": np.where(e & up, amt, 0.0)}).groupby(d).sum()
    out = out.join(sums)

    def mean_of(mask, vals) -> pd.Series:
        m = mask & np.isfinite(vals)
        return pd.Series(vals[m]).groupby(d[m]).mean()

    ok = traded & np.isfinite(pct)
    out["median_pct"] = pd.Series(pct[ok]).groupby(d[ok]).median()
    out["mean_pct"] = mean_of(traded, pct)
    out["prem_zt"] = mean_of(e & pu, pct)             # 昨日涨停溢价（收盘）
    out["prem_zt_open"] = mean_of(e & pu, opct)       # 昨日涨停开盘溢价
    out["prem_lb"] = mean_of(e & pu & (pstreak >= 2), pct)
    out["prem_fb"] = mean_of(e & pfb, pct)
    out["prem_zb"] = mean_of(e & pzb, pct)
    out["prem_dt"] = mean_of(e & pdn, pct)

    names = p["name"].astype(str).to_numpy()
    upr = pd.DataFrame({"date": d[e & up], "streak": streak[e & up], "name": names[e & up]})
    dnr = pd.DataFrame({"date": d[e & dn], "dstreak": dstreak[e & dn], "name": names[e & dn]})
    out["max_h"] = upr.groupby("date")["streak"].max()
    out["max_h"] = out["max_h"].fillna(0).astype("int64")
    out["h2"] = upr[upr["streak"] < upr["date"].map(out["max_h"])].groupby("date")["streak"].max()
    out["top_names"] = _names(upr, "streak", top)
    out["max_dh"] = dnr.groupby("date")["dstreak"].max()
    out["max_dh"] = out["max_dh"].fillna(0).astype("int64")
    out["dtop_names"] = _names(dnr, "dstreak", top)

    out["early_zt"] = np.nan
    if limitup is not None and not limitup.empty:
        lu = limitup[["code", "t1_date", "first_sec"]].dropna(subset=["first_sec"])
        uu = pd.DataFrame({"code": p["full_code"].str[4:].to_numpy()[e & up], "t1_date": d[e & up]})
        m = uu.merge(lu, on=["code", "t1_date"], how="left")
        early = (m["first_sec"] <= EARLY_SEC).groupby(m["t1_date"]).sum()
        covered = out.index.isin(set(lu["t1_date"]))
        out.loc[covered, "early_zt"] = early.reindex(out.index[covered]).fillna(0).to_numpy()

    if index_daily is not None and not index_daily.empty:
        ix = index_daily.pivot_table(index="date", columns="index", values="close").sort_index()
        for name in ("sh", "sz", "gem"):
            if name in ix.columns:
                if name == "sh":
                    out["sh_close"] = ix["sh"].reindex(out.index)
                out[f"{name}_pct"] = ix[name].pct_change().reindex(out.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["zb"] = out["touch"] - out["zt"]
        out["zb_rate"] = (out["zb"] / out["touch"]).where(out["touch"] > 0)
        out["seal_rate"] = (out["zt"] / out["touch"]).where(out["touch"] > 0)
        out["rise_ratio"] = out["n_rise"] / (out["n_rise"] + out["n_fall"]).replace(0, np.nan)
        for k, den in (("promo_fb", "fb"), ("promo_lb", "lb"), ("promo_all", "zt")):
            prev = out[den].shift(1)
            out[k] = (out[f"_{k}"] / prev).where(prev > 0)
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")])

    # 情绪温度
    for k, sign in TEMP_COMPONENTS.items():
        q = rolling_pct(out[k].astype("float64"), window, min_periods)
        out[f"q_{k}"] = q if sign > 0 else 1 - q
    qcols = [f"q_{k}" for k in TEMP_COMPONENTS]
    n_ok = out[qcols].notna().sum(axis=1)
    out["temp"] = (out[qcols].mean(axis=1) * 100).where(n_ok >= len(qcols) - 2)
    out["temp_level"] = out["temp"].map(temp_level)
    return out.rename_axis("date").reset_index()
