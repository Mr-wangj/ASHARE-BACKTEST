"""Jev（TypeSafe System One 模型）竞价买入判断：9:25 竞价结束后逐股评估候选。

遵循 TypeSafe 的 Composite Scoring 模式：数字特征（竞价额 ÷ 昨日成交额、标准化涨幅、封单）由代码
算好写进 state，Jev 只回答“几秒判断”型的模糊问题（抢筹强度、昨日板质量、题材是否主线、是否陷阱、
建议动作），每个问题只评一个维度；代码再按权重合成买入分数并按 confidence 过滤。

- API：POST https://api.typesafe.ai/v1/systemone（urllib 实现，无 SDK 依赖）；
  key 从环境变量读取（默认 TYPESAFE_API_KEY，live_rules.yaml 的 jev.api_key_env 可改）。
- 候选：竞价抢筹 / 竞价涨停（一字候选）/ 昨日涨停 / 一进二候选，按 竞价额 ÷ 昨日 排序取前
  jev.max_candidates 只，逐只并发调用（一只失败不影响其余）。
- 降级：没配 key、调用超时（jev.deadline_seconds）或全部失败时返回 (None, [])，竞价主流程不受影响。
- record_only: true 时预警级别为「低」（不弹通知），先积累样本验证命中率再放开。
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from .engine import Alert, Engine, money

log = logging.getLogger("live")

API_URL = "https://api.typesafe.ai/v1/systemone"

# Score / Choice 的描述要写具体情形（文档建议），不用“较强 / 很强”这类程度词
GRAB_LEVELS = [
    "竞价平淡：竞价额不到昨日全天成交额的 3%，或价格低开，没有资金争夺",
    "竞价温和放量：竞价额达到昨日的 3%~8%，价格小幅高开，参与资金不多",
    "竞价明显抢筹：竞价额达到昨日的 8%~15%，高开幅度明显，资金入场坚决",
    "竞价激烈抢筹：竞价额超过昨日的 15%，大幅高开甚至一字涨停，巨量资金抢筹",
]
BOARD_LEVELS = [
    "昨日涨停板质量差：放量烂板、尾盘勉强回封、盘中多次炸板",
    "昨日涨停板质量一般：正常放量封板，封单中等，偶有开板但快速回封",
    "昨日涨停板质量好：缩量秒板或巨量封单，全天未炸板",
]
ACTION_CHOICES = {
    "strong_buy": "竞价抢筹明确、背景配合，适合按开盘价买入",
    "buy": "偏正面但强度一般，轻仓参与或等回封再介入",
    "watch": "有亮点但不确定因素多，以观望为主",
    "avoid": "疑似出货诱多、位置过高或背景不支持，应回避",
}

DEFAULTS: dict = {
    "enabled": True, "api_key_env": "TYPESAFE_API_KEY", "model": "jev-latest",
    "max_candidates": 30, "concurrency": 6, "timeout_seconds": 20, "deadline_seconds": 150,
    "min_confidence": 0.3,
    "weights": {"grab": 0.3, "board": 0.25, "theme": 0.15, "action": 0.3, "trap": 0.25},
    "buy_threshold": 0.55, "trap_max": 0.4, "require_choice": ["strong_buy"],
    "record_only": True, "max_alerts": 5,
}


def _questions(y_up: bool) -> dict:
    q = {
        "grab": {"type": "score", "instructions": "今日集合竞价数据反映的资金抢筹意愿强度", "criteria": GRAB_LEVELS},
        "is_theme": {"type": "noul",
                     "instructions": "该股所属题材是当前 A 股市场的主线热点方向（近期反复活跃的核心题材，而非冷门或退潮题材）"},
        "trap": {"type": "noul", "instructions": "该股今日竞价表现更像出货或诱多陷阱，而非真实抢筹",
                 "criteria": {"true": "昨日烂板或高位巨量、题材退潮，今日高开放量更像借竞价出逃或诱多",
                              "false": "量价与背景配合自然，不像陷阱"}},
        "action": {"type": "choice", "instructions": "综合竞价表现与昨日背景，今日开盘前对该股的建议动作",
                   "criteria": ACTION_CHOICES},
    }
    if y_up:
        q["board"] = {"type": "score", "instructions": "昨日涨停板的封板质量", "criteria": BOARD_LEVELS}
    return q


def _state(row: pd.Series, mkt: str, seal: dict) -> str:
    y = "涨停第 {s} 板".format(s=int(row["y_streak"])) if row["y_up"] else \
        ("昨日炸板（触及涨停未封住）" if row.get("y_zb") else "昨日未涨停")
    lines = [f"【市场背景】{mkt}",
             f"【个股】{row['name']}（题材：{row['theme'] or '无'}）{row['code']}",
             f"昨日：{y}，全天成交额 {money(row['y_amount'])}，近 20 日最高 {int(row['max_h'])} 板。",
             f"今日集合竞价（9:25 撮合）：涨幅 {row['pct']:+.1%}（涨停幅 {row['limit_pct']:.0%}，即 "
             f"{row['pct'] / row['limit_pct']:.1f} 个涨停），竞价成交额 {money(row['amount'])}"
             f" = 昨日全天成交额的 {row['ratio']:.0%}。"]
    if np.isfinite(row.get("seal_amt", np.nan)) and row["seal_amt"] > 0:
        lines.append(f"竞价涨停，撮合后未成交买单（封单）{money(row['seal_amt'])}。")
    if isinstance(row.get("reason"), str) and row["reason"]:
        lines.append(f"最近一次涨停原因：{row['reason']}。")
    return "\n".join(lines)


def _market_line(engine: Engine) -> str:
    st = engine.ctx.stocks
    zb = int(st["y_zb"].sum())
    return (f"昨日全市场涨停 {int(st['y_up'].sum())} 只、炸板 {zb} 只、最高 {int(st['y_streak'].max())} 板，"
            f"涨停封板率 {(1 - zb / max(1, zb + int(st['y_up'].sum()))) :.0%}。")


def _post(payload: dict, api_key: str, model: str, timeout: float) -> dict:
    req = urllib.request.Request(
        API_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    if "answers" not in out:
        raise RuntimeError(f"响应缺 answers：{str(out)[:120]}")
    return out


def _eval_one(row: pd.Series, mkt: str, seal: dict, api_key: str, jcfg: dict) -> dict:
    """单只股票一次调用（失败重试一次）；返回 answers（不含合成分数）。"""
    payload = {"state": _state(row, mkt, seal), "model": jcfg["model"], "questions": _questions(bool(row["y_up"]))}
    for attempt in (1, 2):
        try:
            return _post(payload, api_key, jcfg["model"], float(jcfg["timeout_seconds"]))["answers"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError, OSError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Jev 调用失败：{str(exc)[:100]}") from exc
            time.sleep(1)


def _combine(ans: dict, y_up: bool, jcfg: dict) -> dict:
    """原子答案 → 合成分数。confidence 低于阈值的因子按中性 0.5 处理（文档建议：不确定时不行动）。"""
    w, minc = jcfg["weights"], jcfg["min_confidence"]

    def score_f(key: str, top: float) -> tuple[float, float]:
        a = ans.get(key) or {}
        v, c = float(a.get("score", 0.0)) / top, float(a.get("confidence", 0.0))
        return (0.5 if c < minc else v), c

    def noul_f(key: str) -> float:
        a = ans.get(key) or {}
        return float(a.get("noul", 0.5))

    action = (ans.get("action") or {}).get("choice", "watch")
    action_f = 0.5 if float((ans.get("action") or {}).get("confidence", 0.0)) < minc else \
        {"strong_buy": 1.0, "buy": 0.8, "watch": 0.5, "avoid": 0.0}.get(action, 0.5)
    grab_f, grab_c = score_f("grab", len(GRAB_LEVELS) - 1)
    board_f, board_c = score_f("board", len(BOARD_LEVELS) - 1) if y_up else (np.nan, np.nan)
    theme_f, trap_f = noul_f("is_theme"), noul_f("trap")

    num = w["grab"] * grab_f + w["theme"] * theme_f + w["action"] * action_f + (w["board"] * board_f if y_up else 0.0)
    den = w["grab"] + w["theme"] + w["action"] + (w["board"] if y_up else 0.0)
    score = num / den - w["trap"] * trap_f
    return {"grab": grab_f, "grab_conf": grab_c, "board": board_f, "board_conf": board_c, "is_theme": theme_f,
            "trap": trap_f, "action": action, "action_f": action_f, "score": round(float(score), 4)}


def _candidates(engine: Engine, jcfg: dict) -> pd.DataFrame:
    """竞价抢筹 / 竞价涨停 / 昨日涨停 / 一进二候选，按 竞价额 ÷ 昨日 排序取前 N 只。"""
    t = engine.auction_tbl
    if not len(t):
        return pd.DataFrame()
    cands = set(engine.ctx.candidates["key_code"].dropna()) if engine.ctx.candidates is not None and len(
        engine.ctx.candidates) else set()
    m = t["grab"] | t["at_limit"] | t["y_up"] | t["code"].isin(cands)
    out = t[m].copy()
    out["ratio"] = out["ratio"].fillna(0.0)
    return out.sort_values("ratio", ascending=False).head(int(jcfg["max_candidates"])).rename(
        columns={"grab": "auc_grab"})      # 避免与 Jev 抢筹分（grab）列名冲突


def judge(engine: Engine, seal: dict, rules: dict, t: datetime,
          deadline: datetime | None = None) -> tuple[pd.DataFrame | None, list[Alert]]:
    """入口：engine.auction() 之后调用。返回 (打分明细表, 买入预警)；未启用 / 无 key / 全部失败 → (None, [])。"""
    jcfg = {**DEFAULTS, **(rules.get("jev") or {})}
    jcfg["weights"] = {**DEFAULTS["weights"], **(jcfg.get("weights") or {})}
    if not jcfg.get("enabled"):
        return None, []
    api_key = os.environ.get(str(jcfg["api_key_env"]), "")
    if not api_key:
        log.warning("Jev 跳过：环境变量 %s 未设置（TypeSafe 控制台 https://console.typesafe.ai/keys 申请）",
                    jcfg["api_key_env"])
        return None, []
    cand = _candidates(engine, jcfg)
    if not len(cand):
        log.info("Jev：没有候选股")
        return None, []
    st = engine.ctx.stocks.reindex(cand["code"])
    for col in ("y_amount", "y_amount5", "max_h", "reason", "y_zb", "limit_pct"):
        if col in st:
            cand[col] = st[col].to_numpy()
    cand["limit_pct"] = cand["limit_pct"].fillna(0.1)
    mkt = _market_line(engine)
    log.info("Jev：%d 只候选，开始逐股评估（模型 %s）", len(cand), jcfg["model"])

    rows: dict[str, dict] = {}
    errors = 0
    cand = cand.reset_index(drop=True)
    code_pos = {c: k for k, c in enumerate(cand["code"])}
    try:
        with ThreadPoolExecutor(max_workers=int(jcfg["concurrency"])) as ex:
            futs: dict[Future, str] = {ex.submit(_eval_one, r, mkt, seal, api_key, jcfg): r["code"]
                                       for _, r in cand.iterrows()}
            try:
                for fut in as_completed(futs, timeout=max(1.0, (deadline - datetime.now()).total_seconds())
                                        if deadline else None):
                    code = futs[fut]
                    try:
                        rows[code] = _combine(fut.result(), bool(cand.at[code_pos[code], "y_up"]), jcfg)
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        log.warning("Jev 评估 %s 失败：%s", code, str(exc)[:120])
            except TimeoutError:
                errors += len(futs) - len(rows)
                log.warning("Jev：到达 deadline（%.0f 秒），已完成 %d/%d 只", jcfg["deadline_seconds"],
                            len(rows), len(futs))
    except Exception as exc:  # noqa: BLE001
        log.exception("Jev 整体失败：%s", exc)
        return None, []
    if not rows:
        log.warning("Jev：%d 只候选全部失败，降级为纯规则", errors)
        return None, []

    done = cand[cand["code"].isin(rows)].reset_index(drop=True)
    info = pd.DataFrame.from_dict(rows, orient="index")
    tbl = done.join(info, on="code")
    tbl["buy"] = (tbl["score"] >= float(jcfg["buy_threshold"])) & (tbl["trap"] <= float(jcfg["trap_max"])) \
        & tbl["action"].isin(jcfg["require_choice"] or [])
    n_buy = int(tbl["buy"].sum())
    engine.stats.update({"jev_n": len(tbl), "jev_fail": errors, "jev_buy": n_buy})
    log.info("Jev 完成：%d 只评分（失败 %d），建议买入 %d 只", len(tbl), errors, n_buy)
    alerts: list[Alert] = []
    lv = "低" if jcfg.get("record_only") else "高"
    hits = tbl[tbl["buy"]].sort_values("score", ascending=False)
    for r in hits.head(int(jcfg["max_alerts"])).itertuples():
        alerts.append(Alert(t, "竞价买入(Jev)", lv,
                            f"{r.name} Jev 建议买入（{r.score:.2f}）",
                            f"动作 {r.action}；抢筹 {r.grab:.2f}、题材主线 {r.is_theme:.2f}、陷阱 {r.trap:.2f}"
                            + (f"、昨日板 {r.board:.2f}" if np.isfinite(r.board) else "")
                            + f"；竞价 {r.pct:+.1%}，竞价额 {money(r.amount)} = 昨日 {r.ratio:.0%}",
                            code=str(r.code), name=str(r.name), sector=r.theme if isinstance(r.theme, str) else None,
                            key=f"jev:{r.code}"))
    keep = ["code", "name", "theme", "pct", "amount", "ratio", "y_up", "y_streak", "auc_grab", "grab", "grab_conf",
            "board", "board_conf", "is_theme", "trap", "action", "action_f", "score", "buy", "reason"]
    return tbl[[c for c in keep if c in tbl.columns]], alerts
