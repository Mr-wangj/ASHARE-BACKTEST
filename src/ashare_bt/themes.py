"""涨停原因 → 题材：拆分标签、按归类表合并成大类、选主题材、配色（同题材同色）。

同花顺“涨停原因类别”形如“AI漫剧+影视院线+北京国资”，先按“+”拆成标签，再按 config/theme_groups.yaml
里的关键词把标签归到大类（如 AI漫剧、AI短剧、微短剧 → 短剧影视）；没匹配到的标签单独成一类。

- 主题材：同花顺把主要涨停原因排在最前，所以按标签顺序取第一个属于归类表“题材”类的大类；
  没有时取第一个题材类标签（未归类的原名），再没有时取第一个大类（事件类，如业绩、国资）。
- 题材内T-1日序号：同一 T 日、同一主题材里按 T-1日序号（T-1 首次涨停时间）排第几；
  题材内T日序号：同一 T 日、同一主题材里按 T日序号（T 日竞价涨幅从高到低）排第几。
- 配色：主题材在当前视图里至少 2 只的“题材”类，按只数取前 10 个，各分配一种固定颜色。
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from .config import PROJECT_ROOT

GROUPS_PATH = PROJECT_ROOT / "config" / "theme_groups.yaml"
KIND_TOPIC, KIND_EVENT = "题材", "事件"

# (背景色, 文字色)：浅底深字；不用红、绿，避免和涨跌颜色混淆；超出的题材显示为灰色
PALETTE: list[tuple[str, str]] = [
    ("#dbeafe", "#1d4ed8"), ("#ede9fe", "#6d28d9"), ("#ffedd5", "#c2410c"), ("#ccfbf1", "#0f766e"),
    ("#fce7f3", "#be185d"), ("#fef9c3", "#a16207"), ("#e0e7ff", "#4338ca"), ("#cffafe", "#0e7490"),
    ("#fae8ff", "#a21caf"), ("#ecfccb", "#4d7c0f"),
]
_SPLIT = re.compile(r"[+＋、,，;；/]")


@dataclass(frozen=True)
class Group:
    name: str
    kind: str
    keywords: tuple[str, ...]


class ThemeMap:
    """标签 → (大类, 类型)。按归类表顺序匹配关键词，先匹配到的优先。"""

    def __init__(self, groups: list[Group]):
        self.groups = groups
        self._cache: dict[str, tuple[str, str]] = {}

    def category(self, tag: str) -> tuple[str, str]:
        if tag not in self._cache:
            hit = next((g for g in self.groups if any(k in tag for k in g.keywords)), None)
            self._cache[tag] = (hit.name, hit.kind) if hit else (tag, KIND_TOPIC)
        return self._cache[tag]

    def kind_of(self, category: str) -> str:
        g = next((g for g in self.groups if g.name == category), None)
        return g.kind if g else KIND_TOPIC

    def known(self, category: str) -> bool:
        """是否为归类表里的大类（而不是没匹配上的原始标签）。"""
        return any(g.name == category for g in self.groups)


def parse_groups(raw: dict) -> list[Group]:
    out = []
    for g in (raw or {}).get("groups", []) or []:
        kws = tuple(str(k).strip() for k in (g.get("keywords") or []) if str(k).strip())
        if g.get("name") and kws:
            out.append(Group(str(g["name"]).strip(), str(g.get("kind") or KIND_TOPIC).strip(), kws))
    return out


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> ThemeMap:
    with open(path, encoding="utf-8") as fh:
        return ThemeMap(parse_groups(yaml.safe_load(fh)))


def load_theme_map(path: Path = GROUPS_PATH) -> ThemeMap:
    """读取归类表（按修改时间缓存，改了文件自动重新加载）；文件不存在时不合并。"""
    if not path.exists():
        return ThemeMap([])
    return _load(str(path), path.stat().st_mtime)


def split_tags(reason) -> list[str]:
    """'医疗器械+创口贴+协议转让' → ['医疗器械', '创口贴', '协议转让']；空值返回 []（保持顺序、去重）。"""
    if reason is None:
        return []
    s = str(reason).strip()
    if not s or s in {"-", "nan", "<NA>", "None"}:
        return []
    out: list[str] = []
    for p in _SPLIT.split(s):
        p = p.strip()
        if p and p not in out:
            out.append(p)
    return out


def categories(tags: list[str], tm: ThemeMap) -> list[str]:
    """标签对应的大类（按标签顺序、去重）。"""
    out: list[str] = []
    for t in tags:
        c = tm.category(t)[0]
        if c not in out:
            out.append(c)
    return out


def category_counts(cat_lists: Iterable[list[str]]) -> Counter:
    """每个大类出现在多少只股票里（同一只股票只算一次）。"""
    c: Counter = Counter()
    for cats in cat_lists:
        c.update(set(cats))
    return c


def main_category(cats: list[str], tm: ThemeMap) -> str | None:
    """cats 按标签顺序排列。依次取：第一个归类表里的题材类 → 第一个题材类（未归类原名）→ 第一个大类。"""
    if not cats:
        return None
    for c in cats:
        if tm.known(c) and tm.kind_of(c) == KIND_TOPIC:
            return c
    return next((c for c in cats if tm.kind_of(c) == KIND_TOPIC), cats[0])


def theme_table(day_df: pd.DataFrame, tm: ThemeMap) -> pd.DataFrame:
    """给一批首板候选算题材信息（应传入所选日期的全部候选，序号才与“竞价红”等筛选无关）。

    需要列：key_t_date、x_t1_rank、x_t_rank、x_t1_reason。返回同索引的表：
    tags、tag_cats（每个标签的大类）、cats、main（主题材）、theme_n（同日同题材只数）、
    theme_rank_t1（题材内T-1日序号）、theme_rank_t（题材内T日序号）。无涨停原因的行 main 为空、序号为 NaN。
    """
    out = pd.DataFrame(index=day_df.index)
    out["key_t_date"] = day_df["key_t_date"]
    out["tags"] = [split_tags(r) for r in day_df["x_t1_reason"]]
    out["tag_cats"] = [[tm.category(t)[0] for t in ts] for ts in out["tags"]]
    out["cats"] = [categories(ts, tm) for ts in out["tags"]]
    out["main"] = [main_category(c, tm) for c in out["cats"]]
    has = out["main"].notna()
    sub = day_df.loc[has]
    keys = [out.loc[has, "key_t_date"], out.loc[has, "main"]]
    out["theme_n"] = sub.groupby(keys)["x_t1_rank"].transform("size")
    out["theme_rank_t1"] = sub.groupby(keys)["x_t1_rank"].rank(method="first")  # T-1 涨停越早越靠前
    out["theme_rank_t"] = sub.groupby(keys)["x_t_rank"].rank(method="first")    # T 日竞价涨幅越高越靠前
    return out


def assign_colors(counts: Counter, tm: ThemeMap, max_themes: int = len(PALETTE),
                  min_count: int = 2) -> dict[str, tuple[str, str]]:
    """出现次数 ≥ min_count 的题材类按次数（并列按名称）排序，前 max_themes 个依次分配颜色。
    非字符串的键（没有涨停原因的股票，pandas 里是 NaN）忽略。"""
    hot = sorted((c for c, n in counts.items() if isinstance(c, str) and n >= min_count and tm.kind_of(c) == KIND_TOPIC),
                 key=lambda c: (-counts[c], c))[:max_themes]
    return {c: PALETTE[i] for i, c in enumerate(hot)}
