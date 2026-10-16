"""每日收盘后自动更新：日线 → 问财涨停明细 → 回测与每日明细（看板「每日一进二明细」的数据）。

    .venv/bin/python scripts/daily_update.py             # 已是最新则几秒内退出
    .venv/bin/python scripts/daily_update.py --dry-run   # 只看是否需要更新
    .venv/bin/python scripts/daily_update.py --force     # 强制完整跑一遍

由 launchd 在工作日 15:45、19:30、22:30、8:50 触发（scripts/launchd/com.ashare.daily-update.plist）；
19:30 那次主要补当天较晚发布的涨停原因；22:30、8:50 只更新舆情（盘后与开盘前的新消息、明日题材预判）。判断逻辑：
- 向同花顺取上证指数的最新交易日（15:30 前只算到上一交易日，不会存入未走完的日 K）；
- 比上次更新完成的交易日新 → 完整更新（日线约 20 分钟 + 问财 + 回测）；
- 是今天且已更新过，但当天问财明细（首次涨停时间、涨停原因、竞价成交额）还没取全 → 只补问财 + 重算；
- 否则（节假日、当天已完成）只更新舆情（scripts/download_news.py，约半分钟；--no-news 可关闭）。
每次行情更新完成后也会更新舆情；舆情失败只记日志，不影响行情数据。
文件锁保证两次任务不重叠；检测到手动运行的下载/回测脚本时也跳过，避免并发查询问财。
日志：data/logs/daily_update.log（每次运行的摘要），data/logs/daily_update_YYYYMMDD.log（各步骤完整输出）。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from datetime import time as dtime

import pandas as pd

from ashare_bt.config import DATA_DIR, FEATURE_DIR, PROJECT_ROOT, RAW_DIR, load_config
from ashare_bt.data.fetch import fetch_index
from ashare_bt.data.wencai_limitup import missing_kinds
from ashare_bt.features.events import board_name, level_name

LOG_DIR = DATA_DIR / "logs"
SUMMARY_LOG = LOG_DIR / "daily_update.log"
STATE_PATH = LOG_DIR / "daily_update_state.json"
LOCK_PATH = DATA_DIR / ".daily_update.lock"
DETAIL_PATH = FEATURE_DIR / "daily_detail.parquet"
MOOD_PATH = FEATURE_DIR / "market_mood.parquet"
KIND_NAMES = {"first_time": "首次涨停时间", "reason": "涨停原因", "auc_amt_t1": "竞价成交额", "auc_amt_t": "次日竞价成交额"}
STEP_NAMES = {"daily": "日线", "limitup": "问财明细", "backtest": "回测与每日明细", "news": "舆情"}
OTHER_JOBS = r"scripts/(download|download_limitup|download_news|run_backtest)\.py"
PRED_DIR = FEATURE_DIR / "theme_predictions"
PROBE_RETRIES = (30, 60, 120)   # 取最新交易日失败时的重试间隔（秒）

log = logging.getLogger("daily_update")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers: list[logging.Handler] = [
        logging.FileHandler(LOG_DIR / f"daily_update_{datetime.now():%Y%m%d}.log", encoding="utf-8")]
    if sys.stdout.isatty():  # 手动运行时同时输出到终端；launchd 下只写日志文件
        handlers.append(logging.StreamHandler(sys.stdout))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def summary(line: str) -> None:
    log.info(line)
    with open(SUMMARY_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}\n")


def notify(title: str, text: str) -> None:
    """macOS 通知（失败忽略）。"""
    script = f"display notification {json.dumps(text, ensure_ascii=False)} with title {json.dumps(title, ensure_ascii=False)}"
    try:
        subprocess.run(["osascript", "-e", script], timeout=10, check=False, capture_output=True)
    except Exception:  # noqa: BLE001
        pass


def latest_trade_day(cfg: dict) -> int:
    """同花顺上证指数的最新交易日（不晚于 download_end：15:30 前为上一交易日）。"""
    end = int(cfg["data"]["download_end"])
    start = int((datetime.strptime(str(end), "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d"))
    df = fetch_index({"sh": cfg["data"]["index_codes"]["sh"]}, start, end)
    return int(df["date"].max())


def probe_latest() -> int:
    """在子进程里查最新交易日：用完即退出，避免本进程与后续下载脚本同时持有同花顺会话。"""
    r = subprocess.run([sys.executable, __file__, "--probe-latest"], cwd=PROJECT_ROOT, capture_output=True,
                       text=True, timeout=300, check=False)
    out = r.stdout.strip().splitlines()
    if r.returncode != 0 or not out or not out[-1].isdigit():
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else
                           f"退出码 {r.returncode}")
    return int(out[-1])


def local_calendar() -> list[int]:
    idx = pd.read_parquet(RAW_DIR / "index_daily.parquet", columns=["index", "date"])
    return sorted(idx.loc[idx["index"] == "sh", "date"].astype("int64").unique().tolist())


def last_done() -> int:
    """上次完整更新到的交易日：优先读状态文件，否则取每日明细里最新的首板日。"""
    if STATE_PATH.exists():
        return int(json.loads(STATE_PATH.read_text(encoding="utf-8"))["trade_day"])
    if DETAIL_PATH.exists():
        return int(pd.read_parquet(DETAIL_PATH, columns=["key_t1_date"])["key_t1_date"].max())
    return 0


def pending_wencai(cal: list[int]) -> dict[int, list[str]]:
    """最新交易日与前一交易日还没取全的问财明细。"""
    out = {cal[-1]: missing_kinds(cal[-1], has_next=False)}
    if len(cal) >= 2:
        out[cal[-2]] = [k for k in missing_kinds(cal[-2], has_next=True) if k == "auc_amt_t"]
    return {d: ks for d, ks in out.items() if ks}


def fmt_pending(pend: dict[int, list[str]]) -> str:
    return "；".join(f"{d} {'、'.join(KIND_NAMES[k] for k in ks)}" for d, ks in pend.items())


def live_monitor_running() -> bool:
    """盘中监控（scripts/live_monitor.py）是否持有它的文件锁。"""
    path = DATA_DIR / ".live_monitor.lock"
    if not path.exists():
        return False
    with open(path, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
    return False


def other_jobs_running() -> list[str]:
    r = subprocess.run(["pgrep", "-fl", OTHER_JOBS], capture_output=True, text=True, check=False)
    mine = {str(os.getpid())}
    return [ln for ln in r.stdout.splitlines() if ln and ln.split()[0] not in mine]


def run_step(name: str, args: list[str], timeout: int, attempts: int = 1) -> None:
    day_log = LOG_DIR / f"daily_update_{datetime.now():%Y%m%d}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    for attempt in range(1, attempts + 1):
        log.info("▶ %s：%s%s", name, " ".join(args), f"（第 {attempt} 次）" if attempt > 1 else "")
        for h in log.handlers:
            h.flush()
        t0 = time.time()
        with open(day_log, "a", encoding="utf-8") as fh:
            try:
                code = subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                      env=env, timeout=timeout, check=False).returncode
            except subprocess.TimeoutExpired:
                code = -1
                log.error("%s 超时（%d 分钟）", name, timeout // 60)
        log.info("◀ %s 结束：退出码 %s，用时 %.1f 分钟", name, code, (time.time() - t0) / 60)
        if code == 0:
            return
        if attempt < attempts:
            time.sleep(120)  # 同花顺偶发失败：稍后重试（下载可续传）
    raise RuntimeError(f"{name} 失败（退出码 {code}），详见 {day_log}")


def run_news() -> str:
    """舆情：快讯、涨停分析、热门 / 人气快照、事件日历与明日题材预判；失败只记日志、不影响行情更新。返回预判摘要。"""
    try:
        run_step(STEP_NAMES["news"], ["scripts/download_news.py"], timeout=1800)
    except Exception as exc:  # noqa: BLE001
        log.warning("舆情更新失败：%s", exc)
        return f"舆情更新失败（{exc}）"
    files = sorted(PRED_DIR.glob("*.parquet"))
    if not files:
        return "舆情已更新"
    p = pd.read_parquet(files[-1]).head(3)
    return "明日题材预判：" + "、".join(f"{t} {x:.0%}" for t, x in zip(p["theme"], p["p_adj"]))


def describe(trade_day: int) -> str:
    """更新结果摘要：最新 T 日各板位的晋级（封板数/候选数），当日首板、二板、三板（次日候选）与问财明细取到的情况。"""
    d = pd.read_parquet(DETAIL_PATH)
    if "key_level" not in d.columns:
        d["key_level"] = 1
    last_t = int(d.loc[~d["key_preview"], "key_t_date"].max())
    t_rows, pv = d[d["key_t_date"] == last_t], d[d["key_preview"]]
    promo = "、".join(f"{level_name(lv)} {int(g['y_t_close_up'].sum())}/{len(g)}" for lv, g in t_rows.groupby("key_level"))
    cand = "、".join(f"{board_name(lv)} {len(g)}" for lv, g in pv.groupby("key_level"))
    mood = ""
    if MOOD_PATH.exists():  # 当日情绪（「情绪监测」页的指标）
        s = pd.read_parquet(MOOD_PATH).set_index("date")
        if trade_day in s.index:
            r = s.loc[trade_day]
            temp = "" if pd.isna(r["temp"]) else f"情绪温度 {r['temp']:.0f}（{r['temp_level']}）、"
            mood = (f"{trade_day} {temp}两市成交 {r['turnover'] / 1e12:.2f} 万亿、涨停 {int(r['zt'])}、跌停 {int(r['dt'])}、"
                    f"炸板率 {r['zb_rate']:.0%}、最高 {int(r['max_h'])} 板；")
    return (f"{mood}T日 {last_t} 晋级 {promo}；{trade_day} {cand} 只（次日候选；首次涨停时间 "
            f"{int(pv['x_t1_first_sec'].notna().sum())}/{len(pv)}、涨停原因 {int(pv['x_t1_reason'].notna().sum())}/{len(pv)}）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="不管是否已是最新，完整跑一遍")
    ap.add_argument("--dry-run", action="store_true", help="只判断是否需要更新")
    ap.add_argument("--notify", action="store_true", help="完成或失败时发 macOS 通知")
    ap.add_argument("--window", type=int, default=5, help="问财明细检查最近多少个交易日（默认 5）")
    ap.add_argument("--no-news", action="store_true", help="不更新舆情（快讯、涨停分析、明日题材预判）")
    ap.add_argument("--probe-latest", action="store_true", help=argparse.SUPPRESS)  # 内部用：打印最新交易日
    args = ap.parse_args()
    if args.probe_latest:
        print(latest_trade_day(load_config()))
        return 0
    setup_logging()

    lock = open(LOCK_PATH, "w")  # noqa: SIM115  进程结束自动释放
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        summary("跳过：另一个更新任务正在运行")
        return 0
    others = other_jobs_running()
    if others:
        summary(f"跳过：检测到正在运行的下载/回测脚本 {others}")
        return 0
    if live_monitor_running() and dtime(9, 10) <= datetime.now().time() <= dtime(15, 5):
        # 同一个同花顺账号同时登录会互相挤掉（10-15 盘中出现过）：盘中让给实时监控，收盘后 15:45 再更新
        summary("跳过：盘中监控正在运行（避免同花顺账号重复登录），收盘后再更新")
        return 0

    latest, err = None, None
    for wait in (*PROBE_RETRIES, None):  # 刚从睡眠唤醒时网络可能还没连上（launchd 唤醒后会立刻补跑错过的任务）
        try:
            latest = probe_latest()
            break
        except Exception as exc:  # noqa: BLE001
            err = exc
            if wait is None:
                break
            log.warning("取最新交易日失败（%s），%d 秒后重试", exc, wait)
            time.sleep(wait)
    if latest is None:
        summary(f"失败：取最新交易日出错（{err}）")
        if args.notify:
            notify("一进二数据更新失败", f"取最新交易日出错：{err}")
        return 1
    done = last_done()
    today = int(datetime.now().strftime("%Y%m%d"))
    pend = pending_wencai(local_calendar()) if done >= latest else {}

    if args.force or done < latest:
        plan = ["daily", "limitup", "backtest"]
        why = "强制更新" if args.force else f"最新交易日 {latest}，上次更新到 {done}"
    elif latest == today and pend:
        plan = ["limitup", "backtest"]
        why = "补当天问财明细：" + fmt_pending(pend)
    else:
        plan, why = [], f"已是最新（交易日 {latest}" + (f"，仍缺 {fmt_pending(pend)}" if pend else "") + "）"
    if args.dry_run:
        steps = [STEP_NAMES[s] for s in plan] + ([] if args.no_news else [STEP_NAMES["news"]])
        print(f"{' → '.join(steps) or '无需更新'}：{why}")
        return 0
    if not plan:
        if args.no_news:
            summary(why + "，无需更新")
            return 0
        t0 = time.time()  # 行情已是最新：只更新舆情（盘后新消息、次日开盘前的消息）
        res = run_news()
        ok = not res.startswith("舆情更新失败")
        summary(f"{why}；{'舆情已更新' if ok else '舆情更新失败'}（{(time.time() - t0) / 60:.1f} 分钟）"
                + (f"：{res}" if ok else f"，下次运行自动补：{res.removeprefix('舆情更新失败')}"))
        return 0 if ok else 1
    log.info("计划：%s（%s）", " → ".join(STEP_NAMES[s] for s in plan), why)

    t0 = time.time()
    try:
        if "daily" in plan:
            run_step("日线", ["scripts/download.py", "--refresh"], timeout=3600, attempts=2)
        cal = local_calendar()
        if cal[-1] != latest:
            log.warning("本地交易日历最新为 %s，与同花顺最新交易日 %s 不一致", cal[-1], latest)
        start = cal[max(0, len(cal) - args.window)]
        if 0 < done < start:  # 中间漏跑了几天：从上次更新到的交易日开始补
            start = done
        run_step("问财涨停明细", ["scripts/download_limitup.py", "--pause", "2", "--start", str(start),
                                  "--end", str(cal[-1])], timeout=3600, attempts=2)
        run_step("回测与每日明细", ["scripts/run_backtest.py"], timeout=1800)
    except Exception as exc:  # noqa: BLE001
        summary(f"失败：{exc}")
        if args.notify:
            notify("一进二数据更新失败", str(exc))
        return 1

    trade_day = local_calendar()[-1]
    left = pending_wencai(local_calendar())
    STATE_PATH.write_text(json.dumps({"trade_day": trade_day, "finished_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                                      "pending": {str(d): ks for d, ks in left.items()}}, ensure_ascii=False),
                          encoding="utf-8")
    msg = describe(trade_day)
    note = f"；仍缺 {fmt_pending(left)}（下次运行自动补）" if left else ""
    news_msg = "" if args.no_news else "；" + run_news()
    steps = [*plan, *([] if args.no_news else ["news"])]
    summary(f"完成（{'、'.join(STEP_NAMES[s] for s in steps)}，{(time.time() - t0) / 60:.1f} 分钟）：{msg}{note}{news_msg}")
    if args.notify:
        notify("一进二数据已更新", msg + note + news_msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
