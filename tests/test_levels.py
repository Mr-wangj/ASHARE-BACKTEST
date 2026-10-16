"""每日明细的板位：首板（一进二）、二板（二进三）、三板（三进四）。"""
import pandas as pd

from ashare_bt.features.events import board_mask, board_name, build_events, level_name
from ashare_bt.features.panel import build_panel, eligible_mask
from ashare_bt.features.sentiment import daily_sentiment

CAL = [20261005, 20261006, 20261007, 20261008, 20261009, 20261012]


def _stock(code, closes):
    prevs = [closes[0]] + closes[:-1]
    return pd.DataFrame({
        "full_code": code, "date": CAL, "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": 1000, "amount": 1e7, "prev_close": prevs,
        "ths_up": False, "ths_bomb": False, "ths_down": False, "ths_warp": False,
    })


def _setup(stocks: dict[str, list[float]], lookback: int = 1):
    daily = pd.concat([_stock(c, v) for c, v in stocks.items()], ignore_index=True)
    codes = list(stocks)
    sec = pd.DataFrame({"full_code": codes, "name": [f"股{i}" for i in range(len(codes))],
                        "board": ["main"] * len(codes), "is_st_now": [False] * len(codes)})
    fs = pd.DataFrame({"full_code": codes, "float_shares": 1e8})
    cfg = {"universe": {"exclude_st": True, "min_list_days": 1, "first_board_lookback": lookback},
           "backtest": {"start": CAL[1], "end": None}, "strategy": {"strength_mode": "lex"}}
    panel = build_panel(daily, sec, fs, CAL, first_board_lookback=lookback)
    sent = daily_sentiment(panel, eligible_mask(panel, exclude_st=True, min_list_days=1))
    return panel, sent, cfg


def _detail(panel, sent, cfg, level):
    return build_events(panel, sent, CAL, cfg, include_last_day=True, include_preview=True, level=level)


def test_names():
    assert [level_name(n) for n in (1, 2, 3)] == ["一进二", "二进三", "三进四"]
    assert [board_name(n) for n in (1, 2, 3)] == ["首板", "二板", "三板"]


def test_level_pools_follow_the_streak():
    # A：10-06 首板、10-07 二板、10-08 三板、10-09 断板；C 从不涨停
    panel, sent, cfg = _setup({"USHA600001": [10.0, 11.0, 12.1, 13.31, 13.0, 13.0],
                               "USHA600003": [10.0, 10.1, 10.0, 10.1, 10.0, 10.1]})
    rows = {lv: _detail(panel, sent, cfg, lv) for lv in (1, 2, 3)}
    got = {lv: list(zip(d["key_code"], d["key_t1_date"], d["key_t_date"], d["y_t_close_up"]))
           for lv, d in rows.items()}
    assert got[1] == [("USHA600001", 20261006, 20261007, True)]    # 一进二成功
    assert got[2] == [("USHA600001", 20261007, 20261008, True)]    # 二进三成功
    assert got[3] == [("USHA600001", 20261008, 20261009, False)]   # 三进四失败
    assert all((d["key_level"] == lv).all() for lv, d in rows.items())
    # 一进二的封板股 = 次日二进三的候选
    sealed = {(c, t) for c, _, t, up in got[1] if up}
    assert sealed == {(c, t1) for c, t1, _, _ in got[2]}


def test_preview_for_higher_levels():
    # 最新交易日 10-12 刚好是 B 的二板：作为次日二进三候选
    panel, sent, cfg = _setup({"USHA600002": [10.0, 10.0, 10.0, 10.0, 11.0, 12.1]})
    d2 = _detail(panel, sent, cfg, 2)
    assert d2["key_preview"].tolist() == [True] and d2["key_t1_date"].tolist() == [20261012]
    assert d2["x_t_rank"].isna().all() and not d2["y_t_close_up"].any()
    assert _detail(panel, sent, cfg, 3).empty                      # 没有三板：空表，不报错


def test_streak_must_start_from_a_first_board():
    # B：10-06 涨停、10-07 未涨停、10-08 和 10-09 连续涨停（连板数 2）
    stocks = {"USHA600002": [10.0, 11.0, 10.0, 11.0, 12.1, 12.0]}
    p1, _, _ = _setup(stocks, lookback=1)
    p3, _, _ = _setup(stocks, lookback=3)
    on = lambda p: p.loc[board_mask(p, 2), "date"].tolist()        # noqa: E731
    assert on(p1) == [20261009]   # 首板口径“前 1 日无涨停”：10-08 是首板，10-09 是二板
    assert on(p3) == []           # 首板口径“前 3 日无涨停”：10-08 不算首板，这串连板不算二板
    assert p3.loc[p3["date"] == 20261009, "streak"].item() == 2
