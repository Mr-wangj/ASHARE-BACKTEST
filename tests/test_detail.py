import math

import numpy as np
import pandas as pd
import pytest

from ashare_bt.data.wencai_limitup import cn_date, extract_field, time_to_seconds
from ashare_bt.features.events import add_cross_section_ranks, add_day_ranks
from ashare_bt.features.limit import BT_SHITI, BT_TZI, BT_YIZI


def test_cn_date_and_time_parse():
    assert cn_date(20261009) == "2026年10月9日"
    assert cn_date(20241231) == "2024年12月31日"
    assert time_to_seconds("09:30:00") == 9 * 3600 + 30 * 60
    assert time_to_seconds("14:46:23") == 14 * 3600 + 46 * 60 + 23
    assert time_to_seconds("9:31") == 9 * 3600 + 31 * 60
    assert math.isnan(time_to_seconds(None)) and math.isnan(time_to_seconds("-"))


def test_extract_field_matches_name_and_date_tag():
    df = pd.DataFrame({
        "full_code": ["USHA600127", "USZA000001", "USHA600241"],
        "code": ["600127", "000001", "600241"],
        "market": ["USHA", "USZA", "USHA"],
        "首次涨停时间[20261009]": ["09:50:45", None, "09:30:00"],
        "分时成交额[20261012 09:25]": [210120170.0, 5.0, 90549888.0],
        "开盘价:不复权[20261012]": [1.0, 2.0, 3.0],
    })
    s = extract_field(df, "首次涨停时间", "20261009")
    assert s.to_dict() == {"600127": "09:50:45", "600241": "09:30:00"}  # 空值被剔除
    assert extract_field(df, "分时成交额", "20261012 09:25")["600241"] == 90549888.0
    assert extract_field(df, "开盘价", "20261012").size == 3  # 去掉“:不复权”后缀
    assert extract_field(df, "首次涨停时间", "20261012").empty  # 日期标签不符（问财默认取了别的日期）
    assert extract_field(None, "首次涨停时间", "20261009").empty


def _pool():
    # 同一 T 日 5 只首板：时间早晚、开盘即涨停的平手、缺失时间
    return pd.DataFrame({
        "key_t_date": [20261012] * 5,
        "key_code": ["USHA600001", "USHA600002", "USHA600003", "USHA600004", "USTM920005"],
        "x_t1_first_sec": [time_to_seconds("10:15:00"), time_to_seconds("09:30:00"),
                           time_to_seconds("09:30:00"), time_to_seconds("14:50:10"), np.nan],
        "x_bt": [BT_SHITI, BT_TZI, BT_YIZI, BT_SHITI, BT_SHITI],
        "x_t1_auc_pct": [0.01, 0.10, 0.10, -0.02, 0.03],
        "x_auc_pct": [0.05, -0.01, 0.10, 0.05, 0.02],
        "x_t_auc_amt": [3e6, 1e6, 9e6, 5e6, np.nan],
        # strength 代理分需要的列
        "x_t1_open_norm": [0.1, 1.0, 1.0, -0.2, 0.1],
        "x_t1_turnover": [0.1, 0.02, 0.01, 0.2, 0.1],
        "x_t1_dd": [0.3, 0.2, 0.0, 0.6, 0.4],
        "x_auc_norm": [0.5, -0.1, 1.0, 0.5, 0.07],
    })


def test_day_ranks_order_by_time_then_board_type():
    ev = add_day_ranks(_pool())
    # 两只 09:30:00：一字（600003）排在 T 字（600002）前；缺失时间（北交所）排最后
    assert ev.set_index("key_code")["x_t1_rank"].to_dict() == {
        "USHA600003": 1, "USHA600002": 2, "USHA600001": 3, "USHA600004": 4, "USTM920005": 5}
    # T日序号：竞价涨幅从高到低；同为 +5% 时竞价成交额大者（600004，500 万）在前
    assert ev.set_index("key_code")["x_t_rank"].to_dict() == {
        "USHA600003": 1, "USHA600004": 2, "USHA600001": 3, "USTM920005": 4, "USHA600002": 5}


def test_time_strength_mode_uses_first_limit_time():
    ev = add_cross_section_ranks(_pool(), "time")
    p = ev.set_index("key_code")["x_P_str"]
    # 有时间的 4 只按 1..4 分位；缺少时间的北交所不参与强弱分位
    assert p["USHA600003"] == 0.0 and p["USHA600004"] == 1.0
    assert p["USHA600001"] == pytest.approx(2 / 3)
    assert math.isnan(p["USTM920005"])
    # Δ = P_str − P_auc：最晚封板的 600004（P_str 1.0）竞价并列第二（P_auc 0.375），弱转强幅度最大
    d = ev.set_index("key_code")["x_delta"]
    assert d["USHA600004"] == pytest.approx(0.625) and d.idxmax() == "USHA600004"


def test_missing_time_never_selected_as_weak_board():
    from ashare_bt.strategy.weak_to_strong import VARIANTS, select
    ev = add_cross_section_ranks(_pool(), "time")
    ev["x_open_limit"] = False
    params = {"p_str_min": 0.5, "p_auc_max": 0.5, "strong_group_max": 0.3, "delta_min": 0.1,
              "auc_norm_lo": 0.0, "auc_norm_hi": 0.9}
    for v in ["base2", "base3", "A", "B", "C"]:
        assert "USTM920005" not in set(ev.loc[select(ev, v, params, min_pool=1), "key_code"]), v
    assert "USTM920005" in set(ev.loc[select(ev, "base1", params, min_pool=1), "key_code"])


def test_ranks_are_per_day():
    a, b = _pool(), _pool()
    b["key_t_date"] = 20261013
    b["key_code"] = b["key_code"] + "X"
    ev = add_day_ranks(pd.concat([a, b], ignore_index=True))
    assert ev.groupby("key_t_date")["x_t1_rank"].apply(sorted).tolist() == [[1, 2, 3, 4, 5]] * 2
