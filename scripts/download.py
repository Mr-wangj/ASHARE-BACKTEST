"""下载全市场日线数据（断点续传）。

    uv run python scripts/download.py            # 首次下载 / 补缺失
    uv run python scripts/download.py --refresh  # 更新到最新交易日（每日收盘后运行，约 20 分钟）

15:30 之前运行时只下载到上一个交易日，避免存入未走完的当日 K 线。
"""
import argparse
import logging
import sys

import pandas as pd

from ashare_bt.config import RAW_DIR, load_config
from ashare_bt.data.fetch import download_all, merge_daily

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="更新到最新交易日")
    args = ap.parse_args()
    cfg = load_config()
    download_all(cfg, refresh=args.refresh)
    sec = pd.read_parquet(RAW_DIR / "securities.parquet")
    daily = merge_daily(sec)
    logging.info("合并完成：%d 行，%d 只，日期 %s ~ %s",
                 len(daily), daily["full_code"].nunique(), daily["date"].min(), daily["date"].max())


if __name__ == "__main__":
    main()
