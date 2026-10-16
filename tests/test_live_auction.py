"""9:25 竞价快照的多来源合并：问财、实时快照（判断是否已更新）、重点股票逐只竞价明细。"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import ashare_bt.live.auction as auction
from ashare_bt.live.context import Context

CODES = ["USHA600001", "USHA600002", "USZA000003", "USZA300004"]
DAY = 20261016


def _ctx():
    st = pd.DataFrame({"name": ["甲", "乙", "丙", "丁"], "board": ["main", "main", "main", "gem"], "limit_pct": [0.1, 0.1, 0.1, 0.2],
                       "y_close": 10.0, "y_open": 9.8, "y_amount": 1e8, "y_up": [True, False, False, False],
                       "y_zb": False, "y_down": False, "y_streak": [1, 0, 0, 0]}, index=CODES)
    cand = pd.DataFrame({"key_code": ["USHA600001"]})
    return Context(day=20261015, stocks=st, concepts=pd.DataFrame(), candidates=cand, pos={c: k for k, c in enumerate(CODES)})


class FakeFeed:
    def __init__(self, wencai=None, rt_fresh=True, quotes_today=True):
        self.wencai, self.rt_fresh, self.quotes_today = wencai, rt_fresh, quotes_today
        self.quote_calls = 0

    def wencai_auction(self, day):
        if self.wencai is None:
            raise RuntimeError("问财限流")
        return self.wencai

    def stocks(self, codes):
        pc = 10.0 if self.rt_fresh else 9.5            # 没更新时还是上一交易日的前收盘价
        op = [10.5, 10.1, 9.9, 11.0] if self.rt_fresh else 9.8
        return pd.DataFrame({"open": op, "prev_close": pc, "up_limit": [11.0, 11.0, 11.0, 12.0],
                             "down_limit": [9.0, 9.0, 9.0, 8.0], "price": op, "amount": 1e8}, index=CODES)

    def auction_last(self, code, day):
        self.quote_calls += 1
        d = datetime(2026, 10, 16, 9, 25) if self.quotes_today else datetime(2026, 10, 15, 9, 25)
        return {"price": 11.0, "matched": 2e6, "bid_left": 5e6, "ask_left": 0.0, "time": d}


def _run(feed, monkeypatch, rounds=3):
    clock = {"t": datetime(2026, 10, 16, 9, 25, 5)}

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["t"]

    def sleep(s):
        clock["t"] += timedelta(seconds=s)

    monkeypatch.setattr(auction, "datetime", FakeDT)
    monkeypatch.setattr(auction.time, "sleep", sleep)
    return auction.auction_snapshot(feed, _ctx(), DAY, clock["t"] + timedelta(seconds=3 * rounds + 1))


def test_wencai_full_coverage(monkeypatch):
    w = (pd.Series({"600001": 5e7, "600002": 1e7, "000003": 2e6, "300004": 3e7}),
         pd.Series({"600001": 11.0, "600002": 10.2, "000003": 9.9, "300004": 11.5}), pd.Series(dtype=float))
    snap, seal = _run(FakeFeed(wencai=w), monkeypatch)
    assert snap.loc["USHA600002", ["price", "amount"]].tolist() == [10.2, 1e7]     # 问财优先
    assert snap.loc["USZA300004", "up_limit"] == 12.0 and "USHA600001" in seal    # 重点股票（昨日涨停）带封单


def test_wencai_down_and_snapshot_stale_keeps_focus_stocks(monkeypatch):
    feed = FakeFeed(wencai=None, rt_fresh=False)
    snap, seal = _run(feed, monkeypatch)
    assert list(seal) == ["USHA600001"]
    assert snap.loc["USHA600001", ["price", "amount"]].tolist() == [11.0, 2.2e7]   # 价格 × 匹配量
    assert np.isnan(snap.loc["USHA600002", "price"])                               # 实时快照没更新：不用它的开盘价
    assert snap.loc["USHA600001", "up_limit"] == 11.0                               # 涨跌停价按昨日收盘价计算
    assert feed.quote_calls == 1                                                    # 已取到的重点股票不重复请求


def test_pct_only_uses_prev_close(monkeypatch):
    w = (pd.Series({c[4:]: 1e7 for c in CODES}), pd.Series(dtype=float),
         pd.Series({"600001": 0.05, "600002": 0.0, "000003": -0.02, "300004": 0.1}))
    snap, _ = _run(FakeFeed(wencai=w), monkeypatch)
    assert snap.loc["USZA000003", "price"] == 9.8 and snap.loc["USZA300004", "price"] == 11.0


def test_holiday_returns_nothing(monkeypatch):
    snap, seal = _run(FakeFeed(wencai=(pd.Series(dtype=float),) * 3, rt_fresh=False, quotes_today=False), monkeypatch)
    assert snap is None and seal == {}
