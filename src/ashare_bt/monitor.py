"""舆情监控：消息分类（config/news_rules.yaml）、题材映射、涨停原因解析，以及“明日题材预判”。

明日题材预判 = 量价基础概率 + 盘后催化：
- 基础概率：按题材（config/theme_groups.yaml 的大类）统计当日涨停家数、昨日涨停家数、最高板、连板家数、10 点前涨停占比、
  连续活跃天数、市场涨停总数，用历史数据拟合逻辑回归，估计“下一交易日该题材涨停 ≥ 3 家”的概率（样本外检验见 train_model）。
- 盘后催化分：收盘后到次日开盘前的快讯按消息定义打分并映射到题材，加上事件日历、问财热门“首次上榜”、人气榜；
  综合分 = logit(基础概率) + catalyst_beta × 催化分。催化部分是规则估计，权重可在消息定义里调整，每天的预判都会保存以便复盘。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import FEATURE_DIR, PROJECT_ROOT
from .themes import KIND_TOPIC, ThemeMap, categories, main_category, split_tags

RULES_PATH = PROJECT_ROOT / "config" / "news_rules.yaml"
MODEL_PATH = FEATURE_DIR / "theme_model.json"
PRED_DIR = FEATURE_DIR / "theme_predictions"
TARGET_N = 3            # “爆发” = 下一交易日该题材涨停 ≥ 3 家
EARLY_SEC = 10 * 3600   # 10:00 前涨停
FRESH_DAYS = 3          # 涨停原因引用的消息在 3 天内 = 当日催化，更早 = 题材属性（背景）


# ================================================================ 消息定义
@dataclass(frozen=True)
class Category:
    name: str
    kind: str
    weight: float
    keywords: tuple[str, ...]
    pattern: re.Pattern = field(compare=False, repr=False)
    exclude: tuple[str, ...] = ()
    exclude_pattern: re.Pattern | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Rules:
    categories: tuple[Category, ...]
    reason_sources: tuple[tuple[str, re.Pattern], ...]
    catalyst_beta: float = 0.5
    important_boost: float = 1.5
    calendar_weight: float = 2.0
    hot_new_weight: float = 1.0
    popularity_weight: float = 0.2
    company_factor: float = 0.3   # 个股公告类消息对题材的催化折算
    baseline_days: int = 10       # 题材消息的“常态”取前 N 个交易日的盘后窗口
    theme_stopwords: tuple[str, ...] = ()   # 新闻→题材映射时不用的泛化词
    ignore_titles: re.Pattern | None = field(default=None, compare=False)   # 行情复盘类标题：不计分


def kw_regex(keywords) -> re.Pattern:
    """关键词 → 正则。短英文（≤4 个字母数字）前后不能紧挨字母，避免 AR 命中 ARM、Meta 命中 Metal。"""
    parts = []
    for k in keywords:
        k = str(k).strip()
        if not k:
            continue
        e = re.escape(k)
        if k.isascii() and k.isalnum() and len(k) <= 4:
            e = rf"(?<![A-Za-z]){e}(?![A-Za-z])"
        parts.append(e)
    return re.compile("|".join(parts), re.I) if parts else re.compile(r"(?!x)x")


def parse_rules(raw: dict) -> Rules:
    cats = tuple(Category(str(c["name"]), str(c.get("kind") or ("利空" if float(c.get("weight", 1)) < 0 else "利好")),
                          float(c.get("weight", 1)), tuple(map(str, c.get("keywords") or [])), kw_regex(c.get("keywords") or []),
                          tuple(map(str, c.get("exclude") or [])), kw_regex(c["exclude"]) if c.get("exclude") else None)
                 for c in raw.get("categories") or [])
    srcs = tuple((str(k), kw_regex(v or [])) for k, v in (raw.get("reason_sources") or {}).items())
    num = {k: float(raw[k]) for k in ("catalyst_beta", "important_boost", "calendar_weight", "hot_new_weight",
                                      "popularity_weight", "company_factor") if raw.get(k) is not None}
    if raw.get("baseline_days") is not None:
        num["baseline_days"] = int(raw["baseline_days"])
    stop = tuple(str(s).strip() for s in raw.get("theme_stopwords") or [] if str(s).strip())
    ignore = kw_regex(raw["ignore_titles"]) if raw.get("ignore_titles") else None
    return Rules(cats, srcs, **num, theme_stopwords=stop, ignore_titles=ignore)


@lru_cache(maxsize=4)
def _load_rules(path: str, mtime: float) -> Rules:
    with open(path, encoding="utf-8") as fh:
        return parse_rules(yaml.safe_load(fh) or {})


def load_rules(path: Path = RULES_PATH) -> Rules:
    """读取消息定义（按修改时间缓存，改了文件自动重新加载）。"""
    return _load_rules(str(path), path.stat().st_mtime)


def classify(text: str, rules: Rules) -> list[tuple[Category, str]]:
    """消息命中的类别（按定义顺序）与命中的关键词；含该类别排除词的消息不算这一类。"""
    out = []
    for c in rules.categories:
        m = c.pattern.search(text or "")
        if m and not (c.exclude_pattern and c.exclude_pattern.search(text or "")):
            out.append((c, m.group(0)))
    return out


def message_score(hits: list[tuple[Category, str]], important: bool, rules: Rules) -> tuple[str | None, float]:
    """一条消息的计分类别与权重：取权重绝对值最大的类别；“重要”快讯乘以 important_boost。"""
    if not hits:
        return None, 0.0
    c = max(hits, key=lambda h: abs(h[0].weight))[0]
    return c.name, c.weight * (rules.important_boost if important else 1.0)


# ================================================================ 题材映射
class Themer:
    """新闻文本 → 题材大类（按 theme_groups.yaml 的关键词）。新闻比涨停原因标签长得多：单字关键词（钢、铜、酒……）
    只在“铜价、铜矿、钢材”这类搭配里算；两个字母以内的英文（AI、AR、VR……）太宽泛，不用；其余短英文要求前后不是字母；
    stop 里的泛化词（消息定义的 theme_stopwords，如“显示”“证券”）不用。"""

    def __init__(self, tm: ThemeMap, stop=()):
        self.tm = tm
        stop = set(stop)
        self.items: list[tuple[str, str, re.Pattern]] = []
        for g in tm.groups:
            pats = []
            for k in g.keywords:
                if (k.isascii() and len(k) <= 2) or k in stop:
                    continue
                if len(k) == 1 and not k.isascii():
                    pats.append(re.escape(k) + "(?=[价矿业材])")
                elif k.isascii() and k.isalnum() and len(k) <= 4:
                    pats.append(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])")
                else:
                    pats.append(re.escape(k))
            self.items.append((g.name, g.kind, re.compile("|".join(pats), re.I) if pats else re.compile(r"(?!x)x")))

    def themes(self, text: str, kinds: tuple[str, ...] = (KIND_TOPIC,)) -> list[str]:
        return [n for n, k, p in self.items if k in kinds and p.search(text or "")]

    def of_block(self, name: str) -> str | None:
        """同花顺概念/行业名 → 题材大类（没有对应大类时返回 None）。"""
        c, kind = self.tm.category(str(name))
        return c if self.tm.known(c) and kind == KIND_TOPIC else None


def reason_theme(reason, tm: ThemeMap) -> str | None:
    return main_category(categories(split_tags(reason), tm), tm)


def assign_themes(ladder: pd.DataFrame, tm: ThemeMap) -> pd.Series:
    """天梯每行的主题材：涨停用当日涨停原因，炸板 / 断板用昨日的。"""
    src = ladder["reason"].fillna(ladder["prv_reason"]) if "prv_reason" in ladder else ladder["reason"]
    m = {r: reason_theme(r, tm) for r in pd.unique(src.dropna())}
    return src.map(m)


def code_theme_map(ladder: pd.DataFrame, themes: pd.Series, days: list[int]) -> dict[str, str]:
    """最近几个交易日涨停过的股票 → 最近一次涨停的主题材（用来把快讯里的关联个股映射到题材）。"""
    m = ladder["is_up"] & ladder["date"].isin(days) & themes.notna()
    sub = pd.DataFrame({"code": ladder.loc[m, "code"], "date": ladder.loc[m, "date"], "theme": themes[m]})
    return sub.sort_values("date").drop_duplicates("code", keep="last").set_index("code")["theme"].to_dict()


# ================================================================ 涨停原因解析（静态分析）
_DISCLAIMER = re.compile(r"[（(]\s*免责声明.*$", re.S)
_SECTION = re.compile(r"(行业原因|公司原因)\s*[：:]")
_ITEM = re.compile(r"(?:^|(?<=[\s。；;]))\d{1,2}、")
_FULL_DATE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_MD_DATE = re.compile(r"(?<![\d年])(\d{1,2})月(\d{1,2})日")
_FRESH_WORDS = ("今日", "今天", "昨日", "昨天", "日前", "近日", "本周", "晚间", "隔夜")
_ROUTINE = ("异常波动", "严重异常", "风险提示", "生产经营正常", "经营情况正常")  # 例行公告：不算催化


@dataclass
class ReasonItem:
    section: str            # 行业 / 公司
    text: str
    source: str | None      # “据……”的来源原文，如“2026年9月3日互动易”
    src_kind: str           # 公告 / 互动易 / 研报 / 政策 / 媒体 / 传闻大V / 产业消息 / 其他
    catalysts: list[str]    # 命中的消息类别（政策、涨价供给……）
    date: int | None        # 引用消息的日期
    age: int | None         # 距当日的天数
    fresh: bool             # 3 天内的消息（当日催化）


def _item_date(text: str, day: date) -> date | None:
    m = _FULL_DATE.search(text)
    try:
        if m:
            return date(int(m[1]), int(m[2]), int(m[3]))
        m = _MD_DATE.search(text)
        if m:
            d = date(day.year, int(m[1]), int(m[2]))
            return d if d <= day + timedelta(days=7) else date(day.year - 1, int(m[1]), int(m[2]))
    except ValueError:
        return None
    return None


def _split_items(body: str) -> list[str]:
    starts = [m.start() for m in _ITEM.finditer(body)]
    if not starts:
        return [body.strip()] if body.strip() else []
    parts = [body[:starts[0]]] + [body[a:b] for a, b in zip(starts, [*starts[1:], len(body)])]
    return [re.sub(r"^\d{1,2}、", "", p.strip()).strip() for p in parts if p.strip()]


def parse_reason(text, day: int, rules: Rules, *, fresh_days: int = FRESH_DAYS) -> list[ReasonItem]:
    """同花顺涨停分析的原因说明 → 条目。分“行业原因 / 公司原因”两段（没有分段时视为公司原因），按“1、2、”拆条，
    识别“据……”的来源类别和引用日期，3 天内的算当日催化。"""
    if text is None or (isinstance(text, float) and math.isnan(text)) or not str(text).strip():
        return []
    body = _DISCLAIMER.sub("", str(text)).strip()
    d0 = datetime.strptime(str(day), "%Y%m%d").date()
    marks = list(_SECTION.finditer(body))
    if marks:
        spans = [(m.group(1)[:2], m.end(), nxt.start() if nxt else len(body))
                 for m, nxt in zip(marks, [*marks[1:], None])]
    else:
        spans = [("公司", 0, len(body))]
    out = []
    for section, a, b in spans:
        for item in _split_items(body[a:b]):
            if len(item) < 4:
                continue
            m = re.match(r"^据(.{2,40}?)[，,。：:]", item)
            source = m.group(1) if m else None
            probe = source or item[:30]
            src_kind = next((k for k, p in rules.reason_sources if p.search(probe)), None)
            if src_kind is None:
                src_kind = "产业消息" if section == "行业" else "其他"
            d = _item_date(source or item, d0)
            age = (d0 - d).days if d else None
            fresh = (age is not None and -1 <= age <= fresh_days) or any(w in item[:30] for w in _FRESH_WORDS)
            fresh = fresh and not any(w in item for w in _ROUTINE)
            out.append(ReasonItem(section, item, source, src_kind, [c.name for c, _ in classify(item, rules)],
                                  int(d.strftime("%Y%m%d")) if d else None, age, fresh))
    return out


# ================================================================ 题材特征与基础模型
FEATURES = {
    "log_n": "今日涨停家数（对数）", "log_prev": "昨日涨停家数（对数）", "max_h": "题材最高板（≤6）",
    "early_share": "10 点前涨停占比", "log_lb": "连板家数（对数）", "hot_days": "连续活跃天数（≤10）",
    "log_mkt": "市场涨停总数（对数）",
}


def theme_daily(ladder: pd.DataFrame, themes: pd.Series) -> pd.DataFrame:
    """每日每个题材：涨停家数 n_up、最高板 max_h、连板家数 n_lb、10 点前涨停 early、一字 n_yizi。"""
    m = ladder["is_up"] & themes.notna()
    up = pd.DataFrame({"date": ladder.loc[m, "date"], "theme": themes[m], "streak": ladder.loc[m, "streak"],
                       "lb": (ladder.loc[m, "streak"] >= 2).astype(int),
                       "early": (ladder.loc[m, "first_sec"] <= EARLY_SEC).astype(int),
                       "yizi": (ladder.loc[m, "bt"] == 3).astype(int)})
    g = up.groupby(["date", "theme"])
    return pd.DataFrame({"n_up": g.size(), "max_h": g["streak"].max(), "n_lb": g["lb"].sum(),
                         "early": g["early"].sum(), "n_yizi": g["yizi"].sum()}).reset_index()


def theme_panel(daily: pd.DataFrame, days: list[int], mkt: pd.Series, topics: list[str], tm: ThemeMap) -> pd.DataFrame:
    """（交易日, 题材）面板：特征 + 标签 y（下一交易日涨停 ≥ TARGET_N）。候选 = 归类表里的全部题材类 + 当日有涨停的其他题材。"""
    themes = sorted(set(topics) | {t for t in daily["theme"].unique() if tm.kind_of(t) == KIND_TOPIC})
    idx = {d: i for i, d in enumerate(days)}
    col = {t: j for j, t in enumerate(themes)}
    d = daily[daily["date"].isin(idx) & daily["theme"].isin(col)]
    shape = (len(days), len(themes))
    mats = {k: np.zeros(shape) for k in ("n_up", "max_h", "n_lb", "early")}
    ii, jj = d["date"].map(idx).to_numpy(), d["theme"].map(col).to_numpy()
    for k in mats:
        mats[k][ii, jj] = d[k].to_numpy()
    n = mats["n_up"]
    prev = np.vstack([np.zeros((1, shape[1])), n[:-1]])
    nxt = np.vstack([n[1:], np.full((1, shape[1]), np.nan)])
    hot = np.zeros(shape)
    for i in range(shape[0]):
        hot[i] = np.where(n[i] >= 2, (hot[i - 1] if i else 0) + 1, 0)
    topic = np.array([t in set(topics) for t in themes])
    cand = (n >= 1) | topic[None, :]
    di, tj = np.nonzero(cand)
    out = pd.DataFrame({
        "date": np.asarray(days)[di], "theme": np.asarray(themes)[tj],
        "n_up": n[di, tj], "n_prev": prev[di, tj], "max_h": mats["max_h"][di, tj], "n_lb": mats["n_lb"][di, tj],
        "early": mats["early"][di, tj], "hot_days": hot[di, tj], "n_next": nxt[di, tj],
    })
    out["mkt"] = out["date"].map(mkt).fillna(0).to_numpy()
    out["y"] = np.where(np.isnan(out["n_next"]), np.nan, (out["n_next"] >= TARGET_N).astype(float))
    return out


def design(df: pd.DataFrame) -> np.ndarray:
    n = df["n_up"].to_numpy(float)
    return np.column_stack([
        np.log1p(n), np.log1p(df["n_prev"].to_numpy(float)), np.minimum(df["max_h"].to_numpy(float), 6),
        df["early"].to_numpy(float) / np.maximum(n, 1), np.log1p(df["n_lb"].to_numpy(float)),
        np.minimum(df["hot_days"].to_numpy(float), 10), np.log1p(df["mkt"].to_numpy(float)),
    ])


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """带 L2 的逻辑回归（牛顿法），X 不含常数列；返回 [截距, 系数...]。"""
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    reg = np.full(A.shape[1], l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y) + reg * w
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(A.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def topk_stats(df: pd.DataFrame, score: str, k: int = 3) -> dict:
    """每日按 score 取前 k 个题材：precision = 其中次日涨停 ≥ TARGET_N 的比例；top_hit = 次日涨停最多的题材在前 k 内的比例。"""
    prec, hit = [], []
    for _, g in df.groupby("date"):
        if g["n_next"].max() <= 0:
            continue
        top = g.nlargest(k, score)
        prec.append(top["y"].mean())
        hit.append(g.loc[g["n_next"].idxmax(), "theme"] in set(top["theme"]))
    return {"precision": float(np.mean(prec)) if prec else float("nan"),
            "top_hit": float(np.mean(hit)) if hit else float("nan"), "days": len(prec)}


def predict_proba(df: pd.DataFrame, model: dict) -> np.ndarray:
    X = (design(df) - np.asarray(model["mean"])) / np.asarray(model["std"])
    w = np.asarray(model["coef"])
    return 1 / (1 + np.exp(-np.clip(w[0] + X @ w[1:], -30, 30)))


def train_model(panel: pd.DataFrame, split: int, l2: float = 1.0) -> dict:
    """样本内（≤ split）拟合、样本外评估；最后用全部样本重拟合作为线上模型。基线 = 按今日涨停家数（并列看最高板）排序。"""
    lab = panel.dropna(subset=["y"])
    tr, te = lab[lab["date"] <= split], lab[lab["date"] > split]

    def fit(df):
        X = design(df)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        return {"mean": mu.tolist(), "std": sd.tolist(),
                "coef": fit_logit((X - mu) / sd, df["y"].to_numpy(float), l2).tolist()}

    m_in = fit(tr)
    te = te.assign(p=predict_proba(te, m_in), base=te["n_up"] + te["max_h"] / 100)
    metrics = {
        "split": int(split), "n_train": int(len(tr)), "n_test": int(len(te)), "pos_rate_test": float(te["y"].mean()),
        "auc": auc(te["y"].to_numpy(), te["p"].to_numpy()), "auc_baseline": auc(te["y"].to_numpy(), te["base"].to_numpy()),
        "top3": topk_stats(te, "p"), "top3_baseline": topk_stats(te, "base"),
    }
    model = fit(lab)
    model.update({"features": list(FEATURES), "feature_names": FEATURES, "target_n": TARGET_N,
                  "trained_through": int(lab["date"].max()), "metrics": metrics})
    return model


def save_model(model: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")


def load_model(path: Path = MODEL_PATH) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ================================================================ 盘后催化与预判
def post_market_window(day: int, next_day: int | None) -> tuple[datetime, datetime]:
    """当日 15:00 到下一交易日 9:25（下一交易日未知时按下一个工作日）。"""
    d = datetime.strptime(str(day), "%Y%m%d")
    if next_day is None:
        n = d + timedelta(days=1)
        while n.weekday() >= 5:
            n += timedelta(days=1)
    else:
        n = datetime.strptime(str(next_day), "%Y%m%d")
    return d.replace(hour=15), n.replace(hour=9, minute=25)


_COMPANY_TITLE = re.compile(r"^[^，。：:\s]{2,10}[：:]")
RECAP = "行情复盘"


def message_level(title: str, tags: str, n_codes: int) -> str:
    """个股 = 公司公告类消息（同花顺标“公告”，或“某公司：……”且只关联一只股票）；其余为行业（题材级）消息。"""
    if "公告" in str(tags or "") or (n_codes == 1 and _COMPANY_TITLE.match(title or "")):
        return "个股"
    return "行业"


def score_messages(flash: pd.DataFrame, rules: Rules, themer: Themer, code_theme: dict[str, str]) -> pd.DataFrame:
    """每条快讯：命中类别 cats、计分类别 category、权重 weight（含“重要”加成）、层级 level（个股 / 行业）、
    对题材的催化 w_theme（个股消息乘以 company_factor）、关联题材 themes（文本关键词 + 关联个股的题材）。"""
    if flash.empty:
        return flash.assign(cats=[], category=[], weight=[], level=[], w_theme=[], themes=[])
    cats, cat, wts, lv, ths = [], [], [], [], []
    for r in flash.itertuples():
        text = f"{r.title} {r.digest}"
        if rules.ignore_titles is not None and rules.ignore_titles.search(r.title or ""):
            hits, name, w = [], RECAP, 0.0   # 行情复盘：不计分
        else:
            hits = classify(text, rules)
            name, w = message_score(hits, bool(r.important), rules)
        codes = list(r.codes) if isinstance(r.codes, (list, np.ndarray)) else []
        t = themer.themes(text)
        for c in codes:
            th = code_theme.get(c)
            if th and th not in t:
                t.append(th)
        cats.append([c.name for c, _ in hits])
        cat.append(name)
        wts.append(w)
        lv.append(message_level(r.title, getattr(r, "tags", ""), len(codes)))
        ths.append(t)
    out = flash.assign(cats=cats, category=cat, weight=wts, level=lv, themes=ths)
    out["w_theme"] = np.where(out["level"] == "个股", out["weight"] * rules.company_factor, out["weight"])
    return out


def theme_points(msgs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """各题材盘后消息的利好点数、利空点数（按 w_theme；一条消息关联多个题材时各计一次）。"""
    if msgs.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    mx = explode_themes(msgs[msgs["w_theme"] != 0])
    if mx.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    pos = mx[mx["w_theme"] > 0].groupby("theme")["w_theme"].sum()
    neg = mx[mx["w_theme"] < 0].groupby("theme")["w_theme"].sum()
    return pos, neg


def abnormal_news(pos: pd.Series, neg: pd.Series, history: list[pd.Series]) -> tuple[pd.Series, pd.Series]:
    """异常消息点数：题材今日占全部利好点数的份额，减去它在前几个交易日的平均份额，再乘以典型窗口的总点数；
    利空点数直接计入。这样消息天天多的大题材（医药、消费……）不会每天都排前面，长假后的窗口也不会因为消息总量大而虚高；
    消息比平时少不算利空（只奖励高出常态的部分）。返回 (异常点数, 常态份额)。没有历史时等于原始点数。"""
    hist = [h for h in history if h.sum() > 0]
    if not hist or pos.sum() <= 0:
        return pos.add(neg, fill_value=0.0), pd.Series(dtype=float)
    base = pd.concat([h / h.sum() for h in hist], axis=1).fillna(0.0).mean(axis=1)
    scale = float(np.median([h.sum() for h in hist] + [pos.sum()]))
    abn = ((pos / pos.sum()).sub(base, fill_value=0.0) * scale).clip(lower=0.0)
    return abn.add(neg, fill_value=0.0), base


def calendar_events(cal: pd.DataFrame, next_day: int, themer: Themer, horizon: int = 3) -> pd.DataFrame:
    """下一交易日起 horizon 天内的事件：关联题材（板块名 + 标题关键词）与距下一交易日的天数。"""
    if cal.empty:
        return cal.assign(themes=[], lag=[])
    n0 = datetime.strptime(str(next_day), "%Y%m%d")
    lag = (pd.to_datetime(cal["date"].astype(str)) - n0).dt.days
    ev = cal[(lag >= 0) & (lag <= horizon)].copy()
    ev["lag"] = lag[ev.index]
    th = []
    for r in ev.itertuples():
        t = [x for x in (themer.of_block(b) for b in (r.blocks if isinstance(r.blocks, (list, np.ndarray)) else [])) if x]
        t += [x for x in themer.themes(r.title) if x not in t]
        th.append(list(dict.fromkeys(t)))
    return ev.assign(themes=th)


def explode_themes(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["themes"].map(len) > 0].explode("themes").rename(columns={"themes": "theme"})


def squash(x):
    """催化点数 → 催化分：sign(x)·ln(1+|x|)，多条消息叠加时边际递减。"""
    return np.sign(x) * np.log1p(np.abs(x))


def predict(day: int, next_day: int | None, *, panel_day: pd.DataFrame, model: dict, news: pd.Series,
            events: pd.DataFrame, hot: pd.DataFrame | None, pop: pd.DataFrame | None, code_theme: dict[str, str],
            themer: Themer, rules: Rules, news_raw: pd.Series | None = None) -> pd.DataFrame:
    """明日题材预判表（每个候选题材一行）：量价特征、基础概率 p_base、各项催化点数、催化分 cat、综合分 score、调整后概率 p_adj。
    news：各题材的盘后消息点数（通常是 abnormal_news 的异常点数）；news_raw：原始点数（只用于显示）。"""
    p = panel_day.copy()
    p["p_base"] = predict_proba(p, model) if len(p) else []
    news = news if news is not None else pd.Series(dtype=float)
    news = news[news.abs() > 1e-9]
    ev = explode_themes(events) if len(events) else pd.DataFrame(columns=["theme", "lag"])
    cal_pts = (ev.assign(w=np.where(ev["lag"] == 0, rules.calendar_weight, rules.calendar_weight / 2))
               .groupby("theme")["w"].sum()) if len(ev) else pd.Series(dtype=float)
    hot_pts = pd.Series(dtype=float)
    if hot is not None and len(hot):
        h = hot.assign(theme=hot["name"].map(themer.of_block))
        new = h[h["hot_tag"].astype(str).str.contains("首次上榜", na=False) & h["theme"].notna()]
        hot_pts = new.groupby("theme").size() * rules.hot_new_weight
    pop_pts = pd.Series(dtype=float)
    if pop is not None and len(pop):
        pt = pop["code"].map(code_theme).dropna()
        pop_pts = pt.value_counts() * rules.popularity_weight
    extra = sorted((set(news.index) | set(cal_pts.index) | set(hot_pts.index)) - set(p["theme"]))
    if extra:  # 只有消息催化、今日没进候选的题材（如未归类的新题材）：量价特征按 0 计
        z = pd.DataFrame({"date": day, "theme": extra})
        for c in ("n_up", "n_prev", "max_h", "n_lb", "early", "hot_days", "n_next", "y"):
            z[c] = 0.0
        z["mkt"] = p["mkt"].iloc[0] if len(p) else 0.0
        z["p_base"] = predict_proba(z, model)
        p = pd.concat([p, z], ignore_index=True)
    for name, s in (("news_pts", news), ("cal_pts", cal_pts), ("hot_pts", hot_pts), ("pop_pts", pop_pts),
                    ("news_raw", news_raw if news_raw is not None else news)):
        p[name] = p["theme"].map(s).fillna(0.0)
    p["cat_pts"] = p[["news_pts", "cal_pts", "hot_pts", "pop_pts"]].sum(axis=1)
    p["cat"] = squash(p["cat_pts"])
    logit = np.log(np.clip(p["p_base"], 1e-6, 1 - 1e-6) / (1 - np.clip(p["p_base"], 1e-6, 1 - 1e-6)))
    p["score"] = logit + rules.catalyst_beta * p["cat"]
    p["p_adj"] = 1 / (1 + np.exp(-p["score"]))
    p["next_day"] = next_day or 0
    return p.sort_values("score", ascending=False).reset_index(drop=True)


def build_theme_panel(ladder: pd.DataFrame, sent: pd.DataFrame, tm: ThemeMap) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """天梯 → (每行主题材, 每日题材统计, 题材面板)。sent：每日情绪（index 为日期，含 zt_count）。"""
    themes = assign_themes(ladder, tm)
    daily = theme_daily(ladder, themes)
    days = sorted(int(d) for d in sent.index if d >= int(ladder["date"].min()))
    topics = [g.name for g in tm.groups if g.kind == KIND_TOPIC]
    return themes, daily, theme_panel(daily, days, sent["zt_count"], topics, tm)


def day_prediction(day: int, next_day: int | None, *, ladder: pd.DataFrame, themes: pd.Series, panel: pd.DataFrame,
                   model: dict, rules: Rules, tm: ThemeMap, until: datetime | None = None) -> dict:
    """某交易日收盘后的预判：读取该日盘后窗口的快讯、事件日历、热门 / 人气快照，返回预判表与中间结果。
    until：只用这个时间之前的消息（回溯评估时用下一交易日 9:25）。"""
    from .data import news  # 避免 data 层在仅做分析时被提前加载

    themer = Themer(tm, rules.theme_stopwords)
    start, end = post_market_window(day, next_day)
    end = min(end, until) if until else end
    days = sorted(int(d) for d in panel["date"].unique())
    recent = [d for d in days if d <= day][-10:]
    code_theme = code_theme_map(ladder, themes, recent)
    msgs = score_messages(news.load_flash(start, end), rules, themer, code_theme)
    pos, neg = theme_points(msgs)
    # 常态：前 baseline_days 个交易日各自盘后窗口的利好点数（只用已有的快讯）
    prev = [d for d in days if d < day][-rules.baseline_days:]
    history = []
    for k, d0 in enumerate(prev):
        d1 = (prev[k + 1] if k + 1 < len(prev) else day)
        h = score_messages(news.load_flash(*post_market_window(d0, d1)), rules, themer, code_theme)
        if len(h):
            history.append(theme_points(h)[0])
    abn, base_share = abnormal_news(pos, neg, history)
    nd = next_day or int(post_market_window(day, None)[1].strftime("%Y%m%d"))
    events = calendar_events(news.load_calendar(), nd, themer)
    hot, pop = news.load_snapshot(day, "hot_blocks"), news.load_snapshot(day, "popularity")
    pred = predict(day, next_day, panel_day=panel[panel["date"] == day], model=model, news=abn, events=events,
                   hot=hot, pop=pop, code_theme=code_theme, themer=themer, rules=rules,
                   news_raw=pos.add(neg, fill_value=0.0))
    pred["news_share_base"] = pred["theme"].map(base_share).fillna(0.0)
    return {"pred": pred, "msgs": msgs, "events": events, "hot": hot, "pop": pop, "window": (start, end),
            "code_theme": code_theme, "history_days": len(history)}


def review_predictions(panel: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """已保存的每日预判 vs 次日实际：前 k 个（按综合分 / 只看基础概率 / 按今日涨停家数）次日涨停 ≥ TARGET_N 的比例。"""
    actual = panel.set_index(["date", "theme"])["n_next"]
    rows = []
    for path in sorted(PRED_DIR.glob("*.parquet")):
        pred = pd.read_parquet(path)
        day = int(path.stem)
        nxt = pred["theme"].map(lambda t: actual.get((day, t), np.nan))
        if nxt.isna().all():
            continue  # 还没有次日结果
        pred = pred.assign(n_next=nxt.fillna(0))
        row = {"date": day, "retro": bool(pred.get("retro", pd.Series([False])).iloc[0])}
        for name, col in (("综合分", "score"), ("基础概率", "p_base"), ("今日涨停家数", "n_up")):
            top = pred.sort_values([col, "max_h"], ascending=False).head(k)
            row[name] = float((top["n_next"] >= TARGET_N).mean())
            row[name + "·题材"] = "、".join(f"{t}({int(n)})" for t, n in zip(top["theme"], top["n_next"]))
        best = pred.loc[pred["n_next"].idxmax()]
        row["次日最强"] = f"{best['theme']}({int(best['n_next'])})"
        rows.append(row)
    return pd.DataFrame(rows)


def save_prediction(day: int, pred: pd.DataFrame, *, retro: bool = False) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{day}.parquet"
    pred.assign(run_at=pd.Timestamp.now().floor("s"), retro=retro).to_parquet(path, index=False)
    return path


def load_prediction(day: int) -> pd.DataFrame | None:
    path = PRED_DIR / f"{day}.parquet"
    return pd.read_parquet(path) if path.exists() else None
