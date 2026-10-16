"""最新交易日的首板作为“次日候选”出现在每日明细里，但不进入回测。"""
import pandas as pd

from ashare_bt.features.events import PREVIEW_T_DATE, build_events
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


def _setup():
    daily = pd.concat([
        _stock("USHA600001", [10.0, 10.0, 10.0, 11.0, 11.5, 11.4]),   # 10-08 首板 → T = 10-09
        _stock("USHA600002", [10.0, 10.0, 10.0, 10.0, 10.0, 11.0]),   # 10-12（最新交易日）首板 → 次日候选
        _stock("USHA600003", [10.0, 10.1, 10.0, 10.1, 10.0, 10.1]),
    ], ignore_index=True)
    sec = pd.DataFrame({"full_code": ["USHA600001", "USHA600002", "USHA600003"], "name": ["甲", "乙", "丙"],
                        "board": ["main"] * 3, "is_st_now": [False] * 3})
    fs = pd.DataFrame({"full_code": sec["full_code"], "float_shares": 1e8})
    cfg = {"universe": {"exclude_st": True, "min_list_days": 1, "first_board_lookback": 1},
           "backtest": {"start": CAL[1], "end": None}, "strategy": {"strength_mode": "lex"}}
    panel = build_panel(daily, sec, fs, CAL, first_board_lookback=1)
    sent = daily_sentiment(panel, eligible_mask(panel, exclude_st=True, min_list_days=1))
    return panel, sent, cfg


def test_preview_rows_only_in_detail():
    panel, sent, cfg = _setup()
    bt = build_events(panel, sent, CAL, cfg)
    assert bt["key_code"].tolist() == ["USHA600001"] and not bt["key_preview"].any()

    detail = build_events(panel, sent, CAL, cfg, include_last_day=True, include_preview=True)
    prev = detail[detail["key_preview"]]
    assert prev["key_code"].tolist() == ["USHA600002"]
    p = prev.iloc[0]
    assert p["key_t1_date"] == 20261012 and p["key_t_date"] == PREVIEW_T_DATE
    assert p["x_t1_rank"] == 1 and pd.isna(p["x_t_rank"]) and pd.isna(p["x_auc_pct"])
    assert not p["y_t_close_up"] and not p["y_t_touch_up"] and not p["x_open_limit"]
    # 有 T 日的行不受影响
    normal = detail[~detail["key_preview"]].iloc[0]
    assert normal["key_t_date"] == 20261009 and normal["x_t_rank"] == 1
