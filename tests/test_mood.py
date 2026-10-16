"""情绪监测日度指标：涨跌停家数、炸板率、高度、晋级率、溢价、亏钱效应、大盘涨跌、情绪温度。"""
import numpy as np
import pandas as pd

from ashare_bt.features.mood import LEVEL_NAMES, market_mood, rolling_pct, temp_level
from ashare_bt.features.panel import build_panel, eligible_mask

CAL = [20261005, 20261006, 20261007, 20261008, 20261009, 20261012]


def _stock(code, closes, highs=None, lows=None):
    prevs = [closes[0]] + closes[:-1]
    return pd.DataFrame({
        "full_code": code, "date": CAL, "open": closes, "high": highs or closes, "low": lows or closes, "close": closes,
        "volume": 1000, "amount": 1e8, "prev_close": prevs,
        "ths_up": False, "ths_bomb": False, "ths_down": False, "ths_warp": False,
    })


def _mood(**kw):
    stocks = {
        "USHA600001": _stock("USHA600001", [10.0, 11.0, 12.1, 13.31, 13.0, 13.0]),             # 甲：首板→2板→3板→断板
        "USHA600002": _stock("USHA600002", [10.0] * 6, highs=[10, 10, 11.0, 10, 10, 10]),        # 乙：10-07 炸板
        "USHA600003": _stock("USHA600003", [10.0, 9.0, 8.1, 8.5, 8.5, 8.5]),                    # 丙：2 连跌后止跌
        "USHA600004": _stock("USHA600004", [10.0, 10.0, 10.0, 9.0, 9.0, 9.0],
                             highs=[10, 10, 10, 11.0, 9, 9]),                                     # 丁：10-08 天地板
        "USHA600005": _stock("USHA600005", [10.0, 11.0, 12.1, 13.31, 11.98, 11.98]),            # 戊：3 板后跌停
        "USHA600006": _stock("USHA600006", [10.0, 11.0, 11.0, 11.0, 11.0, 11.0]),               # ST 己：不计涨停
        "USTM920007": _stock("USTM920007", [10.0, 10.5, 10.5, 10.5, 10.5, 10.5]),               # 北交所：只计大盘
    }
    daily = pd.concat(stocks.values(), ignore_index=True)
    codes = list(stocks)
    sec = pd.DataFrame({"full_code": codes, "name": ["甲", "乙", "丙", "丁", "戊", "己", "庚"],
                        "board": ["main"] * 6 + ["bse"], "is_st_now": [False] * 5 + [True, False]})
    fs = pd.DataFrame({"full_code": codes, "float_shares": 1e8})
    panel = build_panel(daily, sec, fs, CAL, first_board_lookback=1)
    elig = eligible_mask(panel, exclude_st=True, min_list_days=1)
    idx = pd.DataFrame({"date": CAL, "index": "sh", "close": [3000.0, 3030.0, 3000.0, 3000.0, 2970.0, 3000.0]})
    return market_mood(panel, elig, idx, **kw).set_index("date")


def test_limit_counts_heights_and_rates():
    m = _mood()
    assert m.loc[20261006, ["zt", "dt", "fb", "max_h", "max_dh"]].tolist() == [2, 1, 2, 1, 1]   # ST 己不计
    assert m.loc[20261006, "top_names"] == "甲、戊"
    r = m.loc[20261007]
    assert (r["zt"], r["touch"], r["zb"], r["zb_rate"], r["seal_rate"]) == (2, 3, 1, 1 / 3, 2 / 3)
    assert (r["promo_fb"], r["lb"], r["max_dh"], r["dt_lb"], r["dtop_names"]) == (1.0, 2, 2, 1, "丙")
    r = m.loc[20261008]
    assert (r["max_h"], r["b3"], r["tiandi"], r["dt"], r["promo_lb"]) == (3, 2, 1, 1, 1.0)
    assert np.isnan(r["h2"])                                         # 只有 3 板，没有次高板
    r = m.loc[20261009]
    assert (r["damian"], r["high_fall"], r["promo_all"], r["zt"]) == (1, 1, 0.0, 0)   # 戊：3 板后跌停（大面、高位补跌）


def test_premiums_breadth_turnover_and_index():
    m = _mood()
    assert abs(m.loc[20261008, "prem_zt"] - 0.1) < 1e-9                       # 甲、戊 昨日 2 板，今日都 +10%
    assert abs(m.loc[20261008, "prem_dt"] - (8.5 / 8.1 - 1)) < 1e-9          # 丙 昨日跌停
    assert abs(m.loc[20261008, "prem_zb"] - 0.0) < 1e-9                      # 乙 昨日炸板，今日平盘
    expect = np.mean([13.0 / 13.31 - 1, 11.98 / 13.31 - 1])
    assert abs(m.loc[20261009, "prem_zt"] - expect) < 1e-9 and abs(m.loc[20261009, "prem_lb"] - expect) < 1e-9
    r = m.loc[20261006]
    assert (r["n_rise"], r["n_fall"], r["n_stock"], r["n_up5"], r["n_dn5"]) == (4, 1, 7, 4, 1)   # 含 ST、北交所
    assert r["turnover"] == 6e8 and r["turnover_bj"] == 1e8                    # 两市 = 沪深
    assert abs(m.loc[20261006, "sh_pct"] - 0.01) < 1e-12 and m.loc[20261012, "sh_close"] == 3000.0


def test_temperature_and_levels():
    m = _mood(window=4, min_periods=2)
    t = m["temp"].dropna()
    assert len(t) and t.between(0, 100).all()
    assert set(m["temp_level"].dropna()) <= set(LEVEL_NAMES)
    assert [temp_level(x) for x in (0, 19.9, 20, 59.9, 60, 80, 100)] == ["冰点", "冰点", "偏冷", "中性", "偏热", "高潮", "高潮"]
    assert temp_level(float("nan")) is None
    q = rolling_pct(pd.Series([1.0, 3.0, 2.0, 5.0]), 3, 2)
    assert q.round(4).tolist()[1:] == [0.75, 0.5, 0.8333]
    # 跌停家数反向：10-12 没有跌停（近 4 日最少），分位反过来是最高的 1 - 0.5/4
    assert m.loc[20261012, "dt"] == 0 and abs(m.loc[20261012, "q_dt"] - 0.875) < 1e-9
