"""涨停天梯：分层（涨停按连板数，炸板/断板按冲击的那一层）与晋级统计。"""
import pandas as pd

from ashare_bt.features.ladder import (KIND_DK, KIND_DT, KIND_QB, KIND_UP, KIND_ZB, KIND_ZD, build_down_ladder,
                                       build_ladder, ladder_levels)
from ashare_bt.features.panel import build_panel

CAL = [20261005, 20261006, 20261007, 20261008, 20261009, 20261012]


def _stock(code, closes, highs=None, lows=None, opens=None):
    prevs = [closes[0]] + closes[:-1]
    highs, lows, opens = highs or closes, lows or closes, opens or closes
    return pd.DataFrame({
        "full_code": code, "date": CAL, "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": 1000, "amount": 1e7, "prev_close": prevs,
        "ths_up": False, "ths_bomb": False, "ths_down": False, "ths_warp": False,
    })


def _ladder(limitup=None):
    stocks = {
        "USHA600001": _stock("USHA600001", [10.0, 11.0, 12.1, 13.31, 13.0, 13.0]),   # 首板→2板→3板→断板
        "USHA600002": _stock("USHA600002", [10.0] * 6, [10.0, 10.0, 11.0, 10.0, 10.0, 10.0]),  # 10-07 首板炸板
        "USHA600003": _stock("USHA600003", [10.0, 11.0, 12.1, 12.5, 12.5, 12.5],
                             [10.0, 11.0, 12.1, 13.31, 12.5, 12.5]),                  # 2板后冲 3板炸板
    }
    daily = pd.concat(stocks.values(), ignore_index=True)
    codes = list(stocks)
    sec = pd.DataFrame({"full_code": codes, "name": ["甲", "乙", "丙"], "board": ["main"] * 3, "is_st_now": [False] * 3})
    fs = pd.DataFrame({"full_code": codes, "float_shares": 1e8})
    cfg = {"universe": {"exclude_st": True, "min_list_days": 1}}
    panel = build_panel(daily, sec, fs, CAL, first_board_lookback=1)
    return build_ladder(panel, CAL, cfg, limitup=limitup)


def test_rows_and_kinds():
    lad = _ladder()
    got = {(r.date, r.code[-1]): (r.kind, r.row, r.streak, r.prv_streak) for r in lad.itertuples()}
    assert got == {
        (20261006, "1"): (KIND_UP, 1, 1, 0), (20261006, "3"): (KIND_UP, 1, 1, 0),
        (20261007, "1"): (KIND_UP, 2, 2, 1), (20261007, "3"): (KIND_UP, 2, 2, 1),
        (20261007, "2"): (KIND_ZB, 1, 0, 0),        # 首板炸板：第 1 层
        (20261008, "1"): (KIND_UP, 3, 3, 2),
        (20261008, "3"): (KIND_ZB, 3, 0, 2),        # 昨日 2 板，今日冲 3 板炸板：第 3 层
        (20261009, "1"): (KIND_DK, 4, 0, 3),        # 昨日 3 板，今日未触板：第 4 层断板
    }
    dk = lad[(lad["date"] == 20261009)].iloc[0]
    assert abs(dk["close_pct"] - (13.0 / 13.31 - 1)) < 1e-12 and dk["prv_up"]


def test_levels_promotion():
    lad = _ladder()
    lv = ladder_levels(lad, 20261008, 20261007).set_index("height")
    assert lv.index.tolist() == [3]
    assert lv.loc[3, ["n", "fail", "promo_num", "promo_den"]].tolist() == [1, 1, 1, 2]   # 昨日 2 只二板，晋级 1、炸板 1
    lv4 = ladder_levels(lad, 20261009, 20261008).set_index("height")
    assert lv4.loc[4, ["n", "fail", "promo_num", "promo_den"]].tolist() == [0, 1, 0, 1]  # 三板全部断板
    first = ladder_levels(lad, 20261006, None).set_index("height")
    assert first.loc[1, "n"] == 2 and pd.isna(first.loc[1, "promo_den"])


def _down_ladder():
    stocks = {
        # D：10-06 跌停、10-07 再跌停（2 连跌）、10-08 止跌
        "USHA600011": _stock("USHA600011", [10.0, 9.0, 8.1, 8.5, 8.5, 8.5]),
        # E：10-07 盘中触及跌停、收盘拉回（翘板）
        "USHA600012": _stock("USHA600012", [10.0] * 6, lows=[10.0, 10.0, 9.0, 10.0, 10.0, 10.0]),
        # F：10-08 盘中触及涨停、收盘跌停（天地板）
        "USHA600013": _stock("USHA600013", [10.0, 10.0, 10.0, 9.0, 9.0, 9.0], highs=[10.0, 10.0, 10.0, 11.0, 9.0, 9.0],
                             opens=[10.0, 10.0, 10.0, 10.0, 9.0, 9.0]),
    }
    daily = pd.concat(stocks.values(), ignore_index=True)
    codes = list(stocks)
    sec = pd.DataFrame({"full_code": codes, "name": ["丁", "戊", "己"], "board": ["main"] * 3, "is_st_now": [False] * 3})
    fs = pd.DataFrame({"full_code": codes, "float_shares": 1e8})
    panel = build_panel(daily, sec, fs, CAL, first_board_lookback=1)
    return build_down_ladder(panel, {"universe": {"exclude_st": True, "min_list_days": 1}})


def test_down_ladder_rows_and_flags():
    dl = _down_ladder()
    got = {(r.date, r.code[-1]): (r.kind, r.row, r.down_streak, r.prv_down_streak) for r in dl.itertuples()}
    assert got[(20261006, "1")] == (KIND_DT, 1, 1, 0)
    assert got[(20261007, "1")] == (KIND_DT, 2, 2, 1)          # 2 连跌
    assert got[(20261008, "1")] == (KIND_ZD, 3, 0, 2)          # 昨日 2 连跌、今日没触及跌停：止跌，放在第 3 层
    assert got[(20261007, "2")] == (KIND_QB, 1, 0, 0)          # 首次触及跌停后翘板
    d1 = dl[(dl["date"] == 20261006) & (dl["code"] == "USHA600011")].iloc[0]
    assert d1["bt"] == 3 and not d1["tiandi"] and not d1["ths_warp"]   # 开高低收都在跌停价：一字跌停
    f = dl[(dl["date"] == 20261008) & (dl["code"] == "USHA600013")].iloc[0]
    assert f["kind"] == KIND_DT and f["tiandi"] and f["bt"] == 1   # 天地板，实体跌停
    d3 = dl[(dl["date"] == 20261008) & (dl["code"] == "USHA600011")].iloc[0]
    assert abs(d3["dd_high"] - (8.5 / 10.0 - 1)) < 1e-12 and d3["prv_down"]
    lv = ladder_levels(dl, 20261007, 20261006, flag="is_down", streak="down_streak").set_index("height")
    assert lv.loc[2, ["n", "fail", "promo_num", "promo_den"]].tolist() == [1, 0, 1, 1]   # 昨日 1 只跌停，今日续跌
    assert lv.loc[1, ["n", "fail"]].tolist() == [0, 1] and pd.isna(lv.loc[1, "promo_den"])


def test_wencai_fields_today_and_yesterday():
    lu = pd.DataFrame({"code": ["600001", "600003"], "t1_date": [20261008, 20261008],
                       "first_time": ["09:30:00", "10:01:00"], "first_sec": [34200.0, 36060.0],
                       "reason": ["AI应用+短剧", pd.NA], "auc_amt_t1": [1e7, 2e6], "auc_amt_t": [pd.NA, pd.NA]})
    lad = _ladder(lu).set_index(["date", "code"])
    assert lad.loc[(20261008, "USHA600003"), "first_time"] == "10:01:00"          # 炸板股也有首次触板时间
    assert lad.loc[(20261009, "USHA600001"), "prv_reason"] == "AI应用+短剧"        # 断板股用昨日的涨停原因
    assert pd.isna(lad.loc[(20261009, "USHA600001"), "reason"])
