"""盘后舆情数据 + 明日题材预判（看板「舆情监控」页的数据）。

    .venv/bin/python scripts/download_news.py                # 增量：新快讯、当日涨停分析、热门/人气快照、事件日历、预判
    .venv/bin/python scripts/download_news.py --backfill 10  # 首次：回填最近 10 个交易日的快讯与涨停分析，并生成回溯预判
需要先运行 scripts/run_backtest.py（用到涨停天梯 limitup_ladder.parquet 与每日情绪 sentiment.parquet）。
每日自动更新（scripts/daily_update.py）在收盘后、22:30、次日 8:50 各运行一次。
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta

import pandas as pd

from ashare_bt.config import FEATURE_DIR, RAW_DIR, load_config
from ashare_bt.data import news
from ashare_bt.monitor import (build_theme_panel, day_prediction, load_rules, post_market_window, save_model,
                               save_prediction, train_model)
from ashare_bt.themes import load_theme_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("news")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0, help="回填最近 N 个交易日的快讯、涨停分析，并生成回溯预判")
    ap.add_argument("--skip-ztfx", action="store_true", help="不下载涨停分析")
    ap.add_argument("--predict-only", action="store_true", help="不下载，只按现有数据重算模型与预判（改了消息定义后用）")
    args = ap.parse_args()
    cfg = load_config()
    idx = pd.read_parquet(RAW_DIR / "index_daily.parquet")
    calendar = sorted(idx.loc[idx["index"] == "sh", "date"].astype("int64").unique().tolist())
    day = calendar[-1]
    lad = pd.read_parquet(FEATURE_DIR / "limitup_ladder.parquet")
    sent = pd.read_parquet(FEATURE_DIR / "sentiment.parquet").set_index("date").sort_index()
    recent = [d for d in calendar if d >= int(lad["date"].min())][-(args.backfill + 1):]

    if not args.predict_only:
        download(args, day, recent, lad)
    predict_days(cfg, recent, lad, sent)


def download(args, day: int, recent: list[int], lad: pd.DataFrame) -> None:
    # 1) 快讯：从上次抓到的时间（回填时从第一个回填日 15:00）往后，含“重要”栏目
    last = news.latest_flash_time()
    since = datetime.strptime(str(recent[0]), "%Y%m%d").replace(hour=15) if args.backfill else (
        last - timedelta(minutes=30) if last else datetime.strptime(str(day), "%Y%m%d").replace(hour=9))
    flash = news.fetch_flash(since, max_pages=120)
    imp = news.fetch_flash(since, max_pages=40, tag=news.IMPORTANT_TAG)
    added = news.save_flash(pd.concat([flash, imp], ignore_index=True))
    log.info("快讯：%s 起抓到 %d 条（重要 %d），新增 %d 条", f"{since:%m-%d %H:%M}", len(flash), len(imp), added)

    # 2) 涨停分析：当日（回填时最近 N 日）涨停股中还没下载的
    if not args.skip_ztfx:
        for d in recent:
            codes = lad.loc[(lad["date"] == d) & lad["is_up"], "code"].tolist()
            todo = sorted(set(codes) - set(news.load_ztfx(d)["code"]))
            if todo:
                got = news.fetch_ztfx(d, todo)
                news.save_ztfx(d, got)
                log.info("涨停分析 %s：%d 只，取到 %d 只", d, len(todo), len(got))

    # 3) 快照（只能取当前）：问财热门概念/行业、人气榜；事件日历（合并历史）。
    #    只在当日盘后窗口内（下一交易日 9:25 前）更新：开盘后再取会混入下一交易日盘中的信息，导致当日预判“偷看”
    snap_until = post_market_window(day, None)[1]
    if datetime.now() >= snap_until:
        log.info("已过 %s，不再更新 %s 的热门 / 人气快照和事件日历（避免混入下一交易日盘中的信息）",
                 f"{snap_until:%m-%d %H:%M}", day)
        return
    for name, fn in (("hot_blocks", news.fetch_hot_blocks), ("popularity", news.fetch_popularity)):
        try:
            news.save_snapshot(day, name, fn())
        except Exception as exc:  # noqa: BLE001
            log.warning("%s 快照失败：%s", name, exc)
    try:
        news.save_calendar(news.fetch_calendar())
    except Exception as exc:  # noqa: BLE001
        log.warning("事件日历失败：%s", exc)


def predict_days(cfg: dict, recent: list[int], lad: pd.DataFrame, sent: pd.DataFrame) -> None:
    """基础模型（每次用最新数据重训，约 1 秒）与预判：最后一天为明日预判，之前的为回溯预判（用当时到次日开盘前的快讯）。"""
    tm, rules = load_theme_map(), load_rules()
    themes, _daily, panel = build_theme_panel(lad, sent, tm)
    model = train_model(panel, int(cfg["backtest"]["in_sample_end"]))
    save_model(model)
    m = model["metrics"]
    log.info("基础模型：样本外 AUC %.3f（基线 %.3f），每日前 3 命中率 %.1f%%（基线 %.1f%%），次日最强题材在前 3 的比例 %.1f%%（基线 %.1f%%）",
             m["auc"], m["auc_baseline"], m["top3"]["precision"] * 100, m["top3_baseline"]["precision"] * 100,
             m["top3"]["top_hit"] * 100, m["top3_baseline"]["top_hit"] * 100)
    for i, d in enumerate(recent):
        nd = recent[i + 1] if i + 1 < len(recent) else None
        until = post_market_window(d, nd)[1] if nd else None
        res = day_prediction(d, nd, ladder=lad, themes=themes, panel=panel, model=model, rules=rules, tm=tm, until=until)
        save_prediction(d, res["pred"], retro=nd is not None)
        top = res["pred"].head(5)
        log.info("预判 %s%s（盘后消息 %d 条）：%s", d, "（回溯）" if nd else "", len(res["msgs"]),
                 "、".join(f"{t} {p:.0%}" for t, p in zip(top["theme"], top["p_adj"])))


if __name__ == "__main__":
    main()
