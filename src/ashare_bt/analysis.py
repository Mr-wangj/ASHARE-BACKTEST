"""回测脚本与本地看板共用的分析工具。"""
from __future__ import annotations

import numpy as np
import pandas as pd

BOARD_NAMES = {"main": "主板", "gem": "创业板", "star": "科创板", "bse": "北交所"}
AUC_BINS = [-np.inf, -0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, np.inf]
AUC_LABELS = ["<-0.2", "-0.2~0", "0~0.1", "0.1~0.2", "0.2~0.3", "0.3~0.4", "0.4~0.5",
              "0.5~0.6", "0.6~0.7", "0.7~0.85", "≥0.85"]


def segment_masks(ev: pd.DataFrame, in_sample_end: int) -> dict[str, pd.Series]:
    return {
        "全样本": pd.Series(True, index=ev.index),
        "样本内": ev["key_t_date"] <= in_sample_end,
        "样本外": ev["key_t_date"] > in_sample_end,
    }


def tercile(series: pd.Series, labels=("低", "中", "高")) -> pd.Series:
    try:
        return pd.qcut(series, 3, labels=list(labels))
    except ValueError:
        return pd.Series(pd.NA, index=series.index)


def auc_band(ev: pd.DataFrame) -> pd.Series:
    return pd.cut(ev["x_auc_norm"], AUC_BINS, right=False, labels=AUC_LABELS)


def quarter_of(ev: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(ev["key_t_date"].astype(str), format="%Y%m%d").dt.to_period("Q").astype(str)


def quarterly_table(ev: pd.DataFrame, rets: pd.DataFrame, picks: dict[str, pd.Index], rule: str = "r1") -> pd.DataFrame:
    """各口径按季度的样本数、封板率、净收益均值。picks: {口径代号: 入选索引}。"""
    quarter = quarter_of(ev)
    out = None
    for v, idx in picks.items():
        g = pd.DataFrame({"q": quarter.loc[idx], "up": ev.loc[idx, "y_t_close_up"], "r": rets.loc[idx, rule]})
        agg = g.groupby("q").agg(**{f"{v}·样本数": ("r", "size"), f"{v}·封板率": ("up", "mean"),
                                    f"{v}·{rule}均值": ("r", "mean")})
        out = agg if out is None else out.join(agg, how="outer")
    return out.reset_index().rename(columns={"q": "季度"}) if out is not None else pd.DataFrame()
