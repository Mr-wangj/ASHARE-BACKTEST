"""项目路径与配置加载。"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

# 收盘数据稳定后才把“今天”算进下载区间，避免盘中下载到未走完的日 K
CLOSE_READY = (15, 30)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DAILY_DIR = RAW_DIR / "daily"
FEATURE_DIR = DATA_DIR / "features"
REPORT_DIR = PROJECT_ROOT / "reports"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg["data"].get("download_end") is None:
        now = datetime.now()
        end = now if (now.hour, now.minute) >= CLOSE_READY else now - timedelta(days=1)
        cfg["data"]["download_end"] = int(end.strftime("%Y%m%d"))
    return cfg


def ensure_dirs() -> None:
    for d in (RAW_DIR, DAILY_DIR, FEATURE_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
