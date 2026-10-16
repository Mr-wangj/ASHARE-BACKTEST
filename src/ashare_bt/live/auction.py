"""9:25 集合竞价快照：多来源合并（问财 9:25 成交额 / 价格、实时快照、重点股票逐只竞价明细），实时监控在 9:25~9:30 调用。"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from ..features.limit import limit_prices
from .feed import Feed

log = logging.getLogger("live")

AUCTION_MIN_COVER = 0.7   # 取到成交额与价格的股票占比达到 70% 即可计算竞价信号


def auction_snapshot(feed: Feed, ctx, day: int, deadline: datetime) -> tuple[pd.DataFrame | None, dict]:
    """9:25 竞价快照（index = 个股全码；price = 竞价成交价，amount = 竞价成交额，以及当日涨跌停价、前收盘价）。

    三个来源，在 deadline 前多轮尝试、合并：
    1. 问财：全市场 9:25 成交额、成交价 / 涨跌幅（实时字段接口没有分时字段，只能走问财）；
    2. 实时快照：开盘价（与昨日开盘价不同才认为已更新）、当日涨跌停价与前收盘价（前收盘价与昨日收盘价一致才采用）；
    3. 重点股票（昨日涨停 / 炸板 / 跌停、一进二候选）逐只取集合竞价明细：成交价 × 匹配量 = 竞价成交额，同时得到封单。
    价格优先级：重点股票竞价明细 > 问财成交价 > 问财涨跌幅 × 前收盘价 > 实时快照开盘价；成交额：竞价明细 > 问财。
    每轮把各来源取到的只数写进日志（「竞价数据」），便于确认哪个来源可用。返回 (快照, 封单)；都没取到返回 (None, {})。"""
    st, codes = ctx.stocks, ctx.codes
    full = {c[4:]: c for c in codes}
    the_day = datetime.strptime(str(day), "%Y%m%d").date()
    w_amt: dict[str, float] = {}
    w_px: dict[str, float] = {}
    w_pct: dict[str, float] = {}
    rt_open: dict[str, float] = {}
    seal: dict[str, dict] = {}
    limits = None
    focus = st.index[st["y_up"] | st["y_zb"] | st["y_down"]].tolist()
    if ctx.candidates is not None and len(ctx.candidates):
        focus += [c for c in ctx.candidates["key_code"] if c in ctx.pos and c not in focus]
    price = amount = {}
    rnd = 0
    while datetime.now() < deadline:
        rnd += 1
        src = {}
        try:
            a, p, r = feed.wencai_auction(day)
            for s, target in ((a, w_amt), (p, w_px), (r, w_pct)):
                target.update({full[k]: float(v) for k, v in s.items() if k in full and np.isfinite(v)})
            src["问财"] = f"成交额 {len(a)}、价格 {len(p)}、涨跌幅 {len(r)}"
        except Exception as exc:  # noqa: BLE001
            src["问财"] = f"失败（{str(exc)[:60]}）"
        try:
            rt = feed.stocks(codes).reindex(codes)
            pc_ok = (rt["prev_close"] - st["y_close"]).abs() < 0.005
            if pc_ok.mean() > 0.5:   # 当日的前收盘价、涨跌停价已就绪
                limits = rt[["up_limit", "down_limit", "prev_close"]]
            fresh = rt["open"].notna() & ((rt["open"] - st["y_open"]).abs() >= 0.005)
            if fresh.mean() > 0.5:   # 多数股票的开盘价已不是昨日的：认为已更新为当日竞价价
                rt_open = rt.loc[fresh, "open"].astype(float).to_dict()
            src["实时快照"] = f"前收盘价已更新 {pc_ok.mean():.0%}、开盘价已更新 {int(fresh.sum())}"
        except Exception as exc:  # noqa: BLE001
            src["实时快照"] = f"失败（{str(exc)[:60]}）"
        for c in [c for c in focus if c not in seal]:
            if datetime.now() >= deadline:
                break
            try:
                q = feed.auction_last(c, day)
            except Exception:  # noqa: BLE001
                continue
            if q and q.get("time") is not None and q["time"].date() == the_day:
                seal[c] = q
        src["重点股票"] = f"{len(seal)}/{len(focus)}"

        pc = limits["prev_close"] if limits is not None else st["y_close"]
        pct_px = {c: round(float(pc[c]) * (1 + v), 2) for c, v in w_pct.items() if np.isfinite(pc.get(c, np.nan))}
        price = {**rt_open, **pct_px, **w_px, **{c: q["price"] for c, q in seal.items()}}
        amount = {**w_amt, **{c: q["price"] * q["matched"] for c, q in seal.items()}}
        log.info("竞价数据第 %d 轮：成交额 %d 只、价格 %d 只（%s）", rnd, len(amount), len(price),
                 "；".join(f"{k} {v}" for k, v in src.items()))
        if len(amount) >= AUCTION_MIN_COVER * len(codes) and len(price) >= AUCTION_MIN_COVER * len(codes):
            break
        time.sleep(3)
    if not price:
        return None, seal
    if limits is None:   # 实时快照的前收盘价还没更新：按昨日收盘价计算（除权日会有偏差）
        up, dn, _ = limit_prices(st["y_close"], st["board"])
        limits = pd.DataFrame({"up_limit": up, "down_limit": dn, "prev_close": st["y_close"].to_numpy()}, index=codes)
    snap = limits.reindex(codes).assign(price=pd.Series(price, dtype="float64").reindex(codes),
                                        amount=pd.Series(amount, dtype="float64").reindex(codes))
    if snap["price"].notna().mean() < AUCTION_MIN_COVER and not seal:
        return None, seal   # 只有零星价格、也没有重点股票：不足以计算竞价信号
    return snap, seal
