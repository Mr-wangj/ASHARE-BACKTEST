"""S1 一进二·弱转强：三种口径 + 三个基准。

策略函数只能读取 key_/x_ 列（features_only），从机制上杜绝使用 T 日盘中/收盘及之后的信息。

变体：
- base1：全部首板次日竞价买入（剔除开盘即涨停）
- base2：弱板（P_str >= p_str_min）次日竞价买入
- base3：弱板 + 竞价标准化涨幅在 [lo, hi] 内（只有绝对高开、不看相对排名）
- A    ：base3 + 竞价分位 P_auc <= p_auc_max（弱半区里竞价排名靠前）
- B    ：base3 + 竞价涨幅 > 当日强板组（P_str <= strong_group_max）竞价涨幅中位数
- C    ：竞价在 [lo, hi] 内 + 排名跃迁 Δ = P_str - P_auc >= delta_min
"""
from __future__ import annotations

import pandas as pd

VARIANTS = ["base1", "base2", "base3", "A", "B", "C"]
VARIANT_NAMES = {
    "base1": "基准1·全部首板次日",
    "base2": "基准2·弱板次日",
    "base3": "基准3·弱板+竞价区间",
    "A": "口径A·分位反转",
    "B": "口径B·超越强板组",
    "C": "口径C·排名跃迁",
}
SIGNAL_VARIANTS = {"A", "B", "C"}


def features_only(ev: pd.DataFrame) -> pd.DataFrame:
    return ev[[c for c in ev.columns if c.startswith(("key_", "x_"))]]


def select(
    ev: pd.DataFrame,
    variant: str,
    p: dict,
    min_pool: int,
    top_n: int | None = None,
    *,
    boards: list[str] | None = None,
    date_range: tuple[int, int] | None = None,
) -> pd.Index:
    """返回入选行的索引；top_n 仅作用于信号口径（按 Δ 降序取前 N）。

    boards / date_range 在取前 N 之前过滤（截面排名仍基于全市场首板池）。
    """
    f = features_only(ev)
    tradable = ~f["x_open_limit"] & (f["x_pool_n"] >= min_pool)
    weak = f["x_P_str"] >= p["p_str_min"]
    in_band = f["x_auc_norm"].between(p["auc_norm_lo"], p["auc_norm_hi"])

    if variant == "base1":
        mask = tradable
    elif variant == "base2":
        mask = tradable & weak
    elif variant == "base3":
        mask = tradable & weak & in_band
    elif variant == "A":
        mask = tradable & weak & in_band & (f["x_P_auc"] <= p["p_auc_max"])
    elif variant == "B":
        strong = f[f["x_P_str"] <= p["strong_group_max"]]
        med = strong.groupby("key_t_date")["x_auc_norm"].median()  # 含开盘涨停者，反映真实预期
        thr = f["key_t_date"].map(med)
        mask = tradable & weak & in_band & (f["x_auc_norm"] > thr)
    elif variant == "C":
        mask = tradable & in_band & (f["x_delta"] >= p["delta_min"])
    else:
        raise ValueError(f"未知变体 {variant}")

    if boards is not None:
        mask &= f["x_board"].isin(boards)
    if date_range is not None:
        mask &= f["key_t_date"].between(*date_range)

    sel = f[mask.fillna(False)]
    if top_n and variant in SIGNAL_VARIANTS:
        sel = (
            sel.sort_values(["key_t_date", "x_delta"], ascending=[True, False])
            .groupby("key_t_date", sort=False)
            .head(top_n)
        )
    return sel.index
