"""Markdown 报告与图表输出。"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_FONT_CANDIDATES = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "Songti SC", "Hiragino Sans GB"]


def setup_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in _FONT_CANDIDATES:
        if n in names:
            plt.rcParams["font.sans-serif"] = [n]
            break
    plt.rcParams["axes.unicode_minus"] = False


PCT_COLS = {"封板率", "CI下", "CI上", "触板率", "炸板率", "r1胜率", "r2胜率", "r3胜率", "r4胜率",
            "信号封板率", "池内其余封板率", "累计收益", "年化收益", "最大回撤", "占比"}
RET_COLS = {"T日收盘浮盈", "T+1开盘溢价", "r1均值", "r1中位", "r2均值", "r2中位", "r3均值", "r3中位", "r4均值", "r4中位"}


def _fmt(col: str, v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if col in PCT_COLS or col.endswith(("封板率", "胜率")):
        return f"{v:.1%}"
    if col in RET_COLS or col.endswith("均值"):
        return f"{v:+.2%}"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
    return str(v)


def df_to_md(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(c, row[c]) for c in cols) + " |")
    return "\n".join(lines)


def plot_curves(curves: dict[str, pd.DataFrame], title: str, path: Path) -> None:
    setup_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, c in curves.items():
        x = pd.to_datetime(c["date"].astype(str), format="%Y%m%d")
        ax.plot(x, c["nav"], label=label, linewidth=1.3)
    ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bars(df: pd.DataFrame, x: str, ys: list[str], title: str, path: Path) -> None:
    setup_font()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(ys)
    pos = range(len(df))
    for i, y in enumerate(ys):
        ax.bar([p + i * width for p in pos], df[y], width=width, label=y)
    ax.set_xticks([p + width * (len(ys) - 1) / 2 for p in pos])
    ax.set_xticklabels(df[x].astype(str), rotation=30, ha="right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
