"""盘中监控的状态文件与 macOS 通知。看板「盘中监控」页只读这些文件：

data/live/{日期}/
  status.json      运行状态（实时 / 回放、阶段、更新时间、轮询耗时）与市场统计
  alerts.jsonl     预警流（每行一条，追加写入）
  auction.parquet  9:25 竞价表；strategy.parquet 一进二·弱转强实时信号
  sectors.parquet  板块强弱与龙头；limits.parquet 当日触及涨停的股票
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from ..config import PROJECT_ROOT
from .engine import Alert, Engine
from .feed import LIVE_DIR

log = logging.getLogger(__name__)
RULES_PATH = PROJECT_ROOT / "config" / "live_rules.yaml"
STRATEGY_COLS = ["key_code", "key_name", "x_board", "x_t1_rank", "x_pool_n", "x_t1_first_time", "x_bt", "x_t1_reason",
                 "x_t1_auc_pct", "x_auc_pct", "x_auc_norm", "x_t_auc_amt", "x_t_rank", "x_open_limit", "x_P_str",
                 "x_P_auc", "x_delta", "sig_base3", "sig_A", "sig_B", "sig_C", "sig_top"]


_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _numbers(v):
    """PyYAML 把 1.0e7、1e7 这类（指数不带正负号）读成字符串：转回数字，避免手改规则时踩坑。"""
    if isinstance(v, dict):
        return {k: _numbers(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_numbers(x) for x in v]
    if isinstance(v, str) and _NUM.match(v.strip()):
        return float(v)
    return v


def load_rules(path: Path = RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return _numbers(yaml.safe_load(fh))


def day_dir(day: int) -> Path:
    return LIVE_DIR / str(day)


def _atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class Store:
    def __init__(self, day: int, *, fresh: bool = False):
        self.dir = day_dir(day)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.alerts_path = self.dir / "alerts.jsonl"
        if fresh and self.alerts_path.exists():
            self.alerts_path.unlink()

    def add(self, alerts: list[Alert]) -> None:
        if not alerts:
            return
        with open(self.alerts_path, "a", encoding="utf-8") as fh:
            for a in alerts:
                fh.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")

    def write(self, engine: Engine, status: dict) -> None:
        for name, df in (("auction", engine.auction_tbl), ("sectors", engine.sector_tbl), ("limits", engine.limit_tbl)):
            if df is not None and len(df):
                _atomic(df, self.dir / f"{name}.parquet")
        s = engine.strategy_tbl
        if s is not None and len(s):
            _atomic(s[[c for c in STRATEGY_COLS if c in s.columns]], self.dir / "strategy.parquet")
        self.status({**status, "stats": engine.stats})

    def status(self, status: dict) -> None:
        tmp = self.dir / "status.tmp"
        tmp.write_text(json.dumps({**status, "updated_at": datetime.now().isoformat(timespec="seconds")},
                                  ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, self.dir / "status.json")


def read_alerts(day: int) -> pd.DataFrame:
    p = day_dir(day) / "alerts.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["time", "kind", "level", "title", "detail", "code", "name", "sector", "key"])
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if len(df):
        df["time"] = pd.to_datetime(df["time"])
    return df


class Notifier:
    """macOS 通知：只推 notify_levels 里的级别；notify_min_gap 秒内的多条合并成一条，避免刷屏。"""

    def __init__(self, rules: dict, enabled: bool = True):
        self.rules, self.enabled = rules, enabled
        self.pending: list[Alert] = []
        self.last = 0.0

    def push(self, alerts: list[Alert]) -> None:
        lv = set(self.rules.get("notify_levels") or [])
        self.pending += [a for a in alerts if a.level in lv]
        self.flush()

    def flush(self, force: bool = False) -> None:
        if not self.pending or not self.enabled:
            self.pending.clear()
            return
        if not force and time.time() - self.last < float(self.rules.get("notify_min_gap", 20)):
            return
        p, self.pending = self.pending, []
        title = p[0].kind if len(p) == 1 else f"盘中预警 {len(p)} 条"
        text = "；".join(a.title for a in p[:3]) + ("……" if len(p) > 3 else "")
        sound = ' sound name "Glass"' if self.rules.get("notify_sound") else ""
        script = f"display notification {json.dumps(text, ensure_ascii=False)} with title {json.dumps(title, ensure_ascii=False)}{sound}"
        try:
            subprocess.run(["osascript", "-e", script], timeout=10, check=False, capture_output=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("通知失败：%s", exc)
        self.last = time.time()
