import pandas as pd

from ashare_bt.features.panel import build_panel


def _daily(code, closes, prevs, start=20240102):
    dates = list(range(start, start + len(closes)))
    return pd.DataFrame({
        "full_code": code, "date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": 1000, "amount": 1e7, "prev_close": prevs,
        "ths_up": False, "ths_bomb": False, "ths_down": False, "ths_warp": False,
    })


def test_streak_and_first_board():
    # 10 → 11(涨停) → 12.1(涨停) → 12.0 → 13.2(涨停)
    closes = [10.0, 11.0, 12.1, 12.0, 13.2]
    prevs = [9.9, 10.0, 11.0, 12.1, 12.0]
    daily = _daily("USHA600000", closes, prevs)
    cal = daily["date"].tolist()
    sec = pd.DataFrame({"full_code": ["USHA600000"], "name": ["x"], "board": ["main"], "is_st_now": [False]})
    fs = pd.DataFrame({"full_code": ["USHA600000"], "float_shares": [1e8]})
    p = build_panel(daily, sec, fs, cal, first_board_lookback=1)
    assert p["is_up"].tolist() == [False, True, True, False, True]
    assert p["streak"].tolist() == [0, 1, 2, 0, 1]
    assert p["is_first_board"].tolist() == [False, True, False, False, True]
    # 加严：前 3 日无涨停 → 最后一个涨停不再是首板
    p3 = build_panel(daily, sec, fs, cal, first_board_lookback=3)
    assert p3["is_first_board"].tolist() == [False, True, False, False, False]


def test_next_day_fields_and_adjacency():
    closes = [10.0, 11.0, 11.5]
    daily = _daily("USHA600000", closes, [9.9, 10.0, 11.0])
    cal = daily["date"].tolist()
    sec = pd.DataFrame({"full_code": ["USHA600000"], "name": ["x"], "board": ["main"], "is_st_now": [False]})
    fs = pd.DataFrame({"full_code": ["USHA600000"], "float_shares": [1e8]})
    p = build_panel(daily, sec, fs, cal)
    row = p.iloc[1]
    assert row["nxt_date"] == cal[2] and row["nxt_cal_i"] == row["cal_i"] + 1
    assert row["nxt_open"] == 11.5
