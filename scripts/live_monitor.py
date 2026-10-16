"""盘中实时监控与预警（看板「盘中监控」页的数据）。

    .venv/bin/python scripts/live_monitor.py                     # 实时：9:25 前启动，收盘后自动退出（launchd 工作日 9:05 启动）
    .venv/bin/python scripts/live_monitor.py --replay 20261014   # 用当日分时回放一遍（首次会下载全市场分时，约 10 分钟）
    .venv/bin/python scripts/live_monitor.py --once              # 取一次实时快照跑一遍（自检，不写文件）
    .venv/bin/python scripts/live_monitor.py --build-concepts     # 刷新同花顺板块成分（约 2 分钟；实时模式每周自动刷新）

实时流程：盘前用本地数据建背景（昨日涨停 / 连板、成交额、题材、板块成分、一进二候选）→ 9:25:05 取全市场快照算竞价信号
（竞价额 = 9:25 成交额；竞价涨停的再取一次竞价封单）→ 9:30~11:30、13:00~15:00 每 5 秒取全市场快照（约 1 秒），
同时每 15 秒取板块指数、每 30 秒取首次涨停时间、每分钟取同花顺盘口异动 → 预警写入 data/live/{日期}/，高级别弹 macOS 通知。
非交易日（9:25 后的快照仍是上一交易日的数据）自动退出。规则：config/live_rules.yaml（每分钟重新读取）。
"""
from __future__ import annotations

import argparse
import fcntl
import logging
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from ashare_bt.config import DATA_DIR, RAW_DIR, load_config
from ashare_bt.data import ths_client
from ashare_bt.features.limit import limit_prices
from ashare_bt.live.auction import auction_snapshot
from ashare_bt.live.context import build_context, trading_days
from ashare_bt.live.engine import Alert, Engine
from ashare_bt.live.feed import CONCEPTS_PATH, LIVE_DIR, Feed, build_concepts
from ashare_bt.live.store import Notifier, Store, load_rules
from ashare_bt.monitor import Themer
from ashare_bt.themes import load_theme_map

log = logging.getLogger("live")
REPLAY_DIR = LIVE_DIR / "replay"
CONCEPT_MAX_AGE = 7 * 86400
SPECIAL = {"URFI883900": "昨日涨停表现", "URFI883979": "昨日首板表现", "URFI883958": "昨日连板", "URFI883918": "昨日炸板股"}


def setup_logging(day: int, tag: str) -> None:
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers: list[logging.Handler] = [logging.FileHandler(DATA_DIR / "logs" / f"live_{tag}_{day}.log", encoding="utf-8")]
    if sys.stdout.isatty():
        handlers.append(logging.StreamHandler(sys.stdout))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


def at(h: int, m: int, s: int = 0) -> datetime:
    return datetime.now().replace(hour=h, minute=m, second=s, microsecond=0)


def ensure_concepts(feed: Feed, force: bool = False) -> None:
    if force or not CONCEPTS_PATH.exists() or time.time() - CONCEPTS_PATH.stat().st_mtime > CONCEPT_MAX_AGE:
        log.info("刷新同花顺板块成分……")
        c = build_concepts(feed)
        log.info("板块成分：%d 个板块", len(c))


def is_today(snap: pd.DataFrame, ctx) -> bool:
    """快照是不是当日的：成交额与昨日全天成交额几乎相同的股票超过一半 → 还是上一交易日的数据（盘前或休市）。"""
    y = ctx.stocks["y_amount"]
    amt = snap["amount"].reindex(y.index)
    ok = (y > 0) & amt.notna()
    if ok.sum() < 500:
        return False
    same = ((amt[ok] - y[ok]).abs() / y[ok] < 1e-3).mean()
    return same < 0.5


def event_alerts(ev: pd.DataFrame, seen: set[str], themer: Themer) -> list[Alert]:
    out = []
    for r in ev.sort_values("time").itertuples():
        if r.id in seen:
            continue
        seen.add(r.id)
        name = r.block.split(":", 1)[-1]
        th = themer.of_block(name)
        out.append(Alert(r.time, "同花顺异动", "中", r.text, "相关个股：" + "、".join(r.names[:6]),
                         sector=name if not th else f"{name}（{th}）", key=f"同花顺异动:{r.id}"))
    return out


# ====================================================================== 实时
def live(args) -> int:
    today = int(datetime.now().strftime("%Y%m%d"))
    setup_logging(today, "monitor")
    if datetime.now().weekday() >= 5:
        log.info("周末，不监控")
        return 0
    if datetime.now() >= at(15, 0, 30):
        log.info("已收盘，不监控（盘后可用 --replay %s 回放）", today)
        return 0
    lock = open(DATA_DIR / ".live_monitor.lock", "w")  # noqa: SIM115  进程结束自动释放
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("另一个盘中监控正在运行，退出")
        return 0
    rules, cfg, feed = load_rules(), load_config(), Feed()
    ensure_concepts(feed)
    ctx = build_context(rules)
    if ctx.day >= today:
        log.info("本地数据已到 %s（收盘后的数据），无需盘中监控", ctx.day)
        return 0
    engine, store, notifier = Engine(ctx, rules, cfg), Store(today), Notifier(rules)
    themer = Themer(load_theme_map())
    codes, idx_codes = ctx.codes, ctx.concepts["code"].tolist() + list(SPECIAL)
    code_map = {c[4:]: c for c in codes}
    status = {"mode": "live", "day": today, "context_day": ctx.day, "started_at": datetime.now().isoformat(timespec="seconds"),
              "n_stocks": len(codes), "n_concepts": len(ctx.concepts), "n_candidates": len(ctx.candidates)}
    log.info("背景：%s 收盘，股票 %d 只，板块 %d 个，一进二候选 %d 只", ctx.day, len(codes), len(ctx.concepts),
             len(ctx.candidates))

    def phase(p: str, **kw) -> None:
        status.update(phase=p, **kw)
        store.status({**status, "stats": engine.stats})

    # ---- 等竞价结束
    while datetime.now() < at(9, 25, 5):
        phase("集合竞价" if datetime.now() >= at(9, 15) else "盘前准备")
        time.sleep(min(20, max(1, (at(9, 25, 5) - datetime.now()).total_seconds())))
    snap, seal = (auction_snapshot(feed, ctx, today, at(9, 29, 50)) if datetime.now() < at(9, 29, 30)
                  else (None, {}))
    checked = False   # 连续竞价阶段再确认一次快照是当日数据（休市日 9:25 后仍是上一交易日的数据）
    if snap is not None and datetime.now() < at(9, 31):
        up = (snap["price"] - snap["up_limit"]).abs() < 0.005
        for c in [c for c in snap.index[up] if c not in seal][:80]:   # 竞价涨停的再取封单
            try:
                q = feed.auction_last(c, today)
                if q:
                    seal[c] = q
            except Exception as exc:  # noqa: BLE001
                log.warning("竞价封单 %s 失败：%s", c, exc)
        alerts = engine.auction(datetime.now().replace(microsecond=0), snap, seal)
        status["auction_cover"] = (f"竞价数据 {int(snap['price'].notna().sum())}/{len(codes)} 只"
                                   f"（成交额 {int(snap['amount'].notna().sum())} 只）")
        store.add(alerts)
        notifier.push(alerts)
        store.write(engine, {**status, "phase": "竞价结束"})
        log.info("竞价：%d 条预警；竞价涨停 %s 只、抢筹 %s 只", len(alerts), engine.stats.get("auc_limit"),
                 engine.stats.get("auc_grab"))
    else:
        status["note"] = ("启动晚于 9:30，本日不监控竞价" if datetime.now() >= at(9, 29, 30) and snap is None and not seal
                          else "9:30 前没有取到足够的 9:25 竞价数据，本日不监控竞价（详见日志「竞价数据」）")

    # ---- 连续竞价
    seen: set[str] = set()
    last = {"events": 0.0, "stats": 0.0, "index": 0.0, "rules": time.time()}
    first_up: dict[str, datetime] = {}
    index_pct: dict[str, float] = {}
    errors = 0
    while datetime.now() < at(15, 0, 30):
        now = datetime.now()
        if now < at(9, 30):
            phase("竞价结束")
            time.sleep(min(10, (at(9, 30) - now).total_seconds() + 0.5))
            continue
        if at(11, 30, 30) <= now < at(13, 0):
            phase("午休")
            time.sleep(min(60, (at(13, 0) - now).total_seconds() + 0.5))
            continue
        t0 = time.time()
        try:
            if t0 - last["rules"] > 60:
                engine.rules = notifier.rules = rules = load_rules()
                last["rules"] = t0
            snap = feed.stocks(codes)
            if not checked:
                if not is_today(snap, ctx):
                    if datetime.now() >= at(9, 40):
                        phase("非交易日")
                        log.info("9:40 后快照仍是上一交易日的数据：非交易日，退出")
                        return 0
                    time.sleep(10)
                    continue
                checked = True
            if t0 - last["index"] > 15:
                ix = feed.indices(idx_codes)
                index_pct = ix["pct"].dropna().to_dict()
                status["special"] = {SPECIAL[k]: index_pct.get(k) for k in SPECIAL}
                last["index"] = t0
            if t0 - last["stats"] > 30:
                near = snap.index[(snap["high"] - snap["up_limit"]).abs() < 0.005].tolist()
                if near:
                    fu = feed.limit_stats(near)
                    first_up.update({c: v for c, v in fu["first_up"].dropna().items()})
                last["stats"] = t0
            alerts = engine.step(datetime.now().replace(microsecond=0), snap, first_up, index_pct)
            if t0 - last["events"] > rules.get("events_poll_seconds", 60):
                alerts += event_alerts(feed.change_events(code_map), seen, themer)
                last["events"] = t0
            store.add(alerts)
            notifier.push(alerts)
            notifier.flush()
            store.write(engine, {**status, "phase": "连续竞价", "latency": round(time.time() - t0, 2)})
            errors = 0
        except (ths_client.thsdk.AuthenticationError, ths_client.thsdk.NotAuthenticatedError) as exc:
            log.warning("认证失效，重新登录：%s", exc)
            ths_client._AUTHED = False
            time.sleep(5)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.exception("轮询失败（连续 %d 次）：%s", errors, exc)
            phase("出错重试", error=str(exc)[:200])
            time.sleep(min(60, 5 * errors))
            continue
        time.sleep(max(0.5, float(rules.get("poll_seconds", 5)) - (time.time() - t0)))
    notifier.flush(force=True)
    store.write(engine, {**status, "phase": "收盘"})
    log.info("收盘：共 %d 条预警", len(engine.alerts))
    return 0


# ====================================================================== 回放
def load_bars(day: int, codes: list[str], feed: Feed) -> pd.DataFrame:
    """全市场分时（缓存在 data/live/replay/{日期}.parquet，可续传；无分时的股票记一行空值，避免重复请求）。"""
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    path = REPLAY_DIR / f"{day}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["code", "time", "price", "volume", "turnover"])
    have = set(old["code"])
    todo = [c for c in codes if c not in have]
    frames, t0 = [old], time.time()
    for k, c in enumerate(todo, 1):
        try:
            b = feed.intraday_bars(c, day)
        except Exception as exc:  # noqa: BLE001
            log.warning("分时 %s 失败：%s", c, exc)
            continue
        if b is None or b.empty:
            b = pd.DataFrame({"time": [pd.NaT], "price": [np.nan], "volume": [np.nan], "turnover": [np.nan]})
        b.insert(0, "code", c)
        frames.append(b)
        if k % 500 == 0 or k == len(todo):
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            log.info("分时 %d/%d，用时 %.0fs", k, len(todo), time.time() - t0)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def replay(day: int, args) -> int:
    setup_logging(day, "replay")
    rules, cfg, feed = load_rules(), load_config(), Feed()
    cal = trading_days()
    if day not in cal:
        log.error("%s 不在本地交易日历里（请先更新日线）", day)
        return 1
    ensure_concepts(feed)
    ctx = build_context(rules, asof=cal[cal.index(day) - 1])
    engine, store, themer = Engine(ctx, rules, cfg), Store(day, fresh=True), Themer(load_theme_map())
    codes = ctx.codes
    bars = load_bars(day, codes, feed).dropna(subset=["time"])
    P = bars.pivot_table(index="time", columns="code", values="price", aggfunc="last").reindex(columns=codes).ffill()
    A = bars.pivot_table(index="time", columns="code", values="turnover", aggfunc="last").reindex(columns=codes).ffill()
    # 当日涨跌停价：优先用日线的交易所前收（已除权）
    d = pd.read_parquet(RAW_DIR / "daily_all.parquet", filters=[("date", "==", day)], columns=["full_code", "prev_close"])
    pc = d.set_index("full_code")["prev_close"].reindex(codes).fillna(ctx.stocks["y_close"])
    up, dn, _ = limit_prices(pc, ctx.stocks["board"])
    lim = pd.DataFrame({"up_limit": up, "down_limit": dn, "prev_close": pc.to_numpy()}, index=codes)
    times = list(P.index)
    status = {"mode": "replay", "day": day, "context_day": ctx.day, "n_stocks": len(codes),
              "n_concepts": len(ctx.concepts), "n_candidates": len(ctx.candidates), "bars": len(times)}
    log.info("回放 %s：%d 只股票有分时，%d 个时点，板块 %d 个", day, int(P.notna().any().sum()), len(times), len(ctx.concepts))

    # 同花顺盘口异动（只能取到当天的）
    events = feed.change_events({c[4:]: c for c in codes}) if day == int(datetime.now().strftime("%Y%m%d")) else None
    seen: set[str] = set()

    # 9:25 竞价：第一根分时
    t_auc = datetime.strptime(str(day), "%Y%m%d").replace(hour=9, minute=25)
    snap0 = lim.assign(price=P.iloc[0], amount=A.iloc[0], open=P.iloc[0], high=P.iloc[0], low=P.iloc[0])
    at_up = snap0.index[(snap0["price"] - snap0["up_limit"]).abs() < 0.005]
    seal = {}
    if not args.no_seal:
        for c in at_up[:80]:
            try:
                q = feed.auction_last(c, day)
                if q:
                    seal[c] = q
            except Exception as exc:  # noqa: BLE001
                log.warning("竞价封单 %s 失败：%s", c, exc)
    alerts = engine.auction(t_auc, snap0, seal)
    store.add(alerts)
    hi, lo = P.iloc[0].to_numpy(float).copy(), P.iloc[0].to_numpy(float).copy()
    for k in range(1, len(times)):
        t = pd.Timestamp(times[k]).to_pydatetime()
        p = P.iloc[k].to_numpy(float)
        hi, lo = np.fmax(hi, p), np.fmin(lo, p)
        snap = lim.assign(price=p, amount=A.iloc[k].to_numpy(float), open=P.iloc[0].to_numpy(float), high=hi, low=lo)
        new = engine.step(t, snap)
        if events is not None:
            new += event_alerts(events[events["time"] <= t], seen, themer)
        store.add(new)
    if events is not None:
        store.add(event_alerts(events, seen, themer))
    store.write(engine, {**status, "phase": "回放完成"})
    kinds = pd.Series([a.kind for a in engine.alerts]).value_counts().to_dict()
    log.info("回放完成：%d 条预警 %s", len(engine.alerts), kinds)
    return 0


def once(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rules, cfg, feed = load_rules(), load_config(), Feed()
    ensure_concepts(feed)
    ctx = build_context(rules)
    engine = Engine(ctx, rules, cfg)
    t0 = time.time()
    snap = feed.stocks(ctx.codes)
    log.info("快照 %d 只，%.1fs；是否当日数据：%s", len(snap), time.time() - t0, is_today(snap, ctx))
    alerts = engine.step(datetime.now().replace(microsecond=0), snap)
    log.info("统计 %s", {k: v for k, v in engine.stats.items() if k != "time"})
    log.info("预警 %d 条（单个快照没有窗口，只会有涨停 / 跌停类）", len(alerts))
    if len(engine.sector_tbl):
        print(engine.sector_tbl.head(10)[["name", "theme", "n", "avg_pct", "n_up", "space", "pioneer", "zhongjun"]]
              .to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=int, help="用某日分时回放（YYYYMMDD）")
    ap.add_argument("--no-seal", action="store_true", help="回放时不取竞价封单")
    ap.add_argument("--once", action="store_true", help="取一次实时快照自检")
    ap.add_argument("--build-concepts", action="store_true", help="刷新同花顺板块成分")
    args = ap.parse_args()
    if args.build_concepts:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        ensure_concepts(Feed(), force=True)
        return 0
    if args.replay:
        return replay(args.replay, args)
    if args.once:
        return once(args)
    return live(args)


if __name__ == "__main__":
    sys.exit(main())
