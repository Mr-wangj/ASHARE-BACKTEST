"""Jev（TypeSafe System One）竞价买入判断：合成分数、候选选择、预警生成、降级。"""
from datetime import datetime

import pandas as pd
import pytest

import ashare_bt.live.jev as jev_mod
from ashare_bt.live.context import Context
from ashare_bt.live.engine import Engine
from ashare_bt.live.store import load_rules
from tests.test_live import CFG, _ctx, _snap, _t

ANS = {
    "grab": {"type": "score", "score": 2.4, "confidence": 0.9, "probabilities": {}},
    "is_theme": {"type": "noul", "noul": 0.8},
    "trap": {"type": "noul", "noul": 0.2},
    "action": {"type": "choice", "choice": "strong_buy", "confidence": 0.7, "probabilities": {}},
    "board": {"type": "score", "score": 1.5, "confidence": 0.8, "probabilities": {}},
}


def _engine():
    eng = Engine(_ctx(), load_rules(), CFG)
    seal = {"USHA600002": {"price": 11.0, "bid_left": 1e6, "matched": 1e5, "ask_left": 0}}
    eng.auction(_t(9, 25), _snap([10.4, 11.0, 9.3, 10.0, 10.0, 10.0], [3e7, 5e7, 1e6, 1e6, 1e6, 1e6]), seal)
    return eng


def _fake_eval(row, mkt, seal, api_key, jcfg):
    return {k: dict(v) for k, v in ANS.items() if k != "board" or bool(row["y_up"])}


def test_combine_yesterday_limit_up_counts_board():
    jcfg = {**jev_mod.DEFAULTS}
    out = jev_mod._combine(ANS, True, jcfg)
    # 0.3*0.8 + 0.25*0.75 + 0.15*0.8 + 0.3*1.0 - 0.25*0.2 = 0.7975
    assert out["score"] == pytest.approx(0.7975, abs=1e-3)
    out0 = jev_mod._combine(ANS, False, jcfg)                    # 昨日未涨停：board 不计入，权重重分配
    assert pd.isna(out0["board"]) and out0["score"] == pytest.approx((0.3 * 0.8 + 0.15 * 0.8 + 0.3) / 0.75 - 0.05)


def test_combine_low_confidence_is_neutral():
    jcfg = {**jev_mod.DEFAULTS, "min_confidence": 0.95}
    out = jev_mod._combine(ANS, True, jcfg)                      # 所有 confidence < 0.95 → 中性
    assert out["grab"] == 0.5 and out["action_f"] == 0.5


def test_judge_skips_without_key_or_when_disabled(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    tbl, alerts = jev_mod.judge(_engine(), {}, load_rules(), _t(9, 25))
    assert tbl is None and alerts == []
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    rules = load_rules()
    rules["jev"]["enabled"] = False
    assert jev_mod.judge(_engine(), {}, rules, _t(9, 25)) == (None, [])


def test_judge_scores_candidates_and_alerts(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(jev_mod, "_eval_one", _fake_eval)
    eng = _engine()
    tbl, alerts = jev_mod.judge(eng, {}, load_rules(), _t(9, 25))
    assert list(tbl["code"]) == ["USHA600002", "USHA600001", "USHA600003"]     # 一字(ratio .17) + 抢筹(.15) + 昨日涨停，按 ratio 排序
    assert tbl["buy"].all() and tbl["score"].iloc[0] == pytest.approx(0.7975, abs=1e-3)   # 首行是一字（昨日涨停，含 board）
    assert eng.stats["jev_n"] == 3 and eng.stats["jev_buy"] == 3
    assert [a.kind for a in alerts] == ["竞价买入(Jev)"] * 3
    assert alerts[0].level == "低" and alerts[0].key == f"jev:{alerts[0].code}"   # record_only: true → 低级不弹通知
    assert "建议买入" in alerts[0].title


def test_judge_high_level_when_not_record_only(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(jev_mod, "_eval_one", _fake_eval)
    rules = load_rules()
    rules["jev"]["record_only"] = False
    _, alerts = jev_mod.judge(_engine(), {}, rules, _t(9, 25))
    assert alerts and alerts[0].level == "高"


def test_judge_degrades_when_all_fail(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")

    def boom(row, mkt, seal, api_key, jcfg):
        raise RuntimeError("Jev 调用失败：超时")

    monkeypatch.setattr(jev_mod, "_eval_one", boom)
    tbl, alerts = jev_mod.judge(_engine(), {}, load_rules(), _t(9, 25))
    assert tbl is None and alerts == []


def test_state_contains_numbers(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    eng = _engine()
    cand = jev_mod._candidates(eng, jev_mod.DEFAULTS)
    st = eng.ctx.stocks.reindex(cand["code"])
    for col in ("y_amount", "max_h", "reason", "limit_pct"):
        cand[col] = st[col].to_numpy()
    row = cand[cand["code"] == "USHA600001"].iloc[0]
    s = jev_mod._state(row, jev_mod._market_line(eng), {})
    assert "USHA600001" in s and "+4.0%" in s and "15%" in s and "竞价成交额" in s
