"""下载 T-1 首板的涨停明细（首次涨停时间、涨停原因、T-1/T 日 9:25 竞价成交额），可续传。

用法：
    uv run python scripts/download_limitup.py                 # 覆盖回测区间全部 T-1 日
    uv run python scripts/download_limitup.py --start 20260901 --end 20261009
需要先运行 scripts/download.py（用到日线与交易日历）。
"""
import argparse
import logging
import sys

import pandas as pd

from ashare_bt.config import RAW_DIR, load_config
from ashare_bt.data.wencai_limitup import PACER, download_limitup, merge_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("limitup")


def main() -> None:
    cfg = load_config()
    idx = pd.read_parquet(RAW_DIR / "index_daily.parquet")
    calendar = sorted(idx.loc[idx["index"] == "sh", "date"].astype("int64").unique().tolist())
    first_t = next(d for d in calendar if d >= cfg["backtest"]["start"])
    default_start = calendar[calendar.index(first_t) - 1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=default_start, help="T-1 日起点 YYYYMMDD")
    ap.add_argument("--end", type=int, default=calendar[-1],
                    help="T-1 日终点 YYYYMMDD（默认最新交易日：当天收盘后即可取当天涨停明细）")
    ap.add_argument("--kinds", nargs="*", default=None, help="first_time reason auc_amt_t1 auc_amt_t")
    ap.add_argument("--pause", type=float, default=None, help="查询间隔秒数（问财会限流，后台补数据建议 3~5）")
    args = ap.parse_args()
    if args.pause is not None:
        PACER.base = PACER.pause = args.pause

    # close / prev_close 用于自算涨停：同花顺涨停标记收盘后可能还没更新，覆盖率分母并上自算涨停
    daily = pd.read_parquet(RAW_DIR / "daily_all.parquet", columns=["full_code", "date", "ths_up", "close", "prev_close"])
    rep = download_limitup(calendar, args.start, args.end, daily, args.kinds)
    merged = merge_cache()
    if not rep.empty:
        summ = rep.groupby("kind")["coverage"].agg(["count", "mean", "min"])
        log.info("覆盖率汇总：\n%s", summ.to_string())
    log.info("合并完成：%d 行 → %s", len(merged), RAW_DIR / "limitup_detail.parquet")


if __name__ == "__main__":
    main()
