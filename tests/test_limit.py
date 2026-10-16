import numpy as np
import pandas as pd
import pytest

from ashare_bt.features.limit import (
    BT_NONE, BT_SHITI, BT_TZI, BT_YIZI, add_limit_columns, limit_price_exact, limit_prices,
    round_half_up_cent,
)


@pytest.mark.parametrize(
    "prev,pct,expected",
    [
        (10.00, 0.10, 11.00),
        (1.15, 0.10, 1.27),     # 1.265 → 1.27（浮点存为 1.2649999 也必须进位）
        (12.35, 0.10, 13.59),   # 13.585 → 13.59
        (9.99, 0.20, 11.99),    # 11.988
        (3.33, 0.30, 4.33),     # 4.329
        (1.15, -0.10, 1.04),    # 跌停 1.035 → 1.04
        (193.31, 0.20, 231.97),  # 同花顺 2024-10-08 实际涨停价
    ],
)
def test_known_limit_prices(prev, pct, expected):
    assert limit_price_exact(prev, pct) == pytest.approx(expected)
    assert round_half_up_cent(prev * (1 + pct)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "prev,expected_up,expected_dn",
    [
        (9.19, 11.94, 6.44),   # 铁大科技 2024-10-28：11.947 截断为 11.94；6.433 向上取整 6.44
        (3.83, 4.97, 2.69),    # 百甲科技：4.979 → 4.97
        (10.39, 13.50, 7.28),  # 旺成科技：13.507 → 13.50
    ],
)
def test_bse_limit_prices(prev, expected_up, expected_dn):
    up, dn, _ = limit_prices(pd.Series([prev]), pd.Series(["bse"]))
    assert up[0] == pytest.approx(expected_up) and dn[0] == pytest.approx(expected_dn)
    assert limit_price_exact(prev, 0.3, "bse") == pytest.approx(expected_up)
    assert limit_price_exact(prev, -0.3, "bse") == pytest.approx(expected_dn)


def test_vectorized_matches_decimal():
    rng = np.random.default_rng(0)
    prev = np.round(rng.uniform(0.5, 2000, 20_000), 2)
    for board, pct in [("main", 0.10), ("gem", 0.20), ("star", 0.20), ("bse", 0.30)]:
        up, dn, _ = limit_prices(pd.Series(prev), pd.Series([board] * len(prev)))
        exp_up = np.array([limit_price_exact(p, pct, board) for p in prev])
        exp_dn = np.array([limit_price_exact(p, -pct, board) for p in prev])
        assert np.abs(up - exp_up).max() < 1e-9, board
        assert np.abs(dn - exp_dn).max() < 1e-9, board


def test_board_type():
    df = pd.DataFrame({
        "board": ["main"] * 4 + ["gem"],
        "prev_close": [10.0, 10.0, 10.0, 10.0, 193.31],
        "open":  [11.0, 11.0, 10.2, 10.5, 231.97],
        "high":  [11.0, 11.0, 11.0, 11.0, 231.97],
        "low":   [11.0, 10.6, 10.1, 10.3, 214.97],
        "close": [11.0, 11.0, 11.0, 10.9, 231.97],
    })
    out = add_limit_columns(df)
    assert out["board_type"].tolist() == [BT_YIZI, BT_TZI, BT_SHITI, BT_NONE, BT_TZI]
    assert out["is_up"].tolist() == [True, True, True, False, True]
    assert out["touch_up"].tolist() == [True, True, True, True, True]


def test_st_day_not_counted_as_limit_up():
    # 主板 ST 日按 5% 涨停：收盘 10.50 达不到 10% 涨停价 11.00
    df = pd.DataFrame({"board": ["main"], "prev_close": [10.0], "open": [10.1], "high": [10.5],
                       "low": [10.0], "close": [10.5]})
    assert not add_limit_columns(df)["is_up"].iloc[0]
