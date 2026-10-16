"""交易成本：佣金（双边）、印花税（卖出）、过户费（双边）、滑点（买卖各一次）。

注：佣金最低 5 元未计入，等价于假设单笔成交额 >= 2 万元。
"""
from __future__ import annotations

import numpy as np


def net_return(buy_px, sell_px, costs: dict) -> np.ndarray:
    slip = costs["slippage"]
    buy = np.asarray(buy_px, dtype="float64") * (1 + slip)
    sell = np.asarray(sell_px, dtype="float64") * (1 - slip)
    cost_in = 1 + costs["commission"] + costs["transfer_fee"]
    cost_out = 1 - costs["commission"] - costs["stamp_duty"] - costs["transfer_fee"]
    return sell * cost_out / (buy * cost_in) - 1
