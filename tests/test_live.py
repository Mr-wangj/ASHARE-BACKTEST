"""盘中信号引擎：竞价信号、个股异动、板块集体异动（含重合板块合并）、龙头、冷却、一进二实时信号。"""
from datetime import datetime

import numpy as np
import pandas as pd

from ashare_bt.live.context import Context
from ashare_bt.live.engine import Engine
from ashare_bt.live.store import load_rules

CFG = {"strategy": {"strength_mode": "time", "p_str_min": 0.5, "p_auc_max": 0.2, "strong_group_max": 0.3,
                    "delta_min": 0.4, "auc_norm_lo": 0.2, "auc_norm_hi": 0.7, "top_n": 2},
       "universe": {"min_pool": 8}}
CODES = ["USHA600001", "USHA600002", "USHA600003", "USHA600004", "USHA600005", "USHA600006"]


def _ctx(candidates=None) -> Context:
    st = pd.DataFrame({
        "name": ["甲", "乙", "丙", "丁", "戊", "己"], "board": "main", "limit_pct": 0.1, "y_close": 10.0,
        "y_amount": [2e8, 3e8, 1e8, 1e8, 1e8, 1e8], "y_amount5": 1e8,
        "y_up": [False, True, True, False, False, False], "y_streak": [0, 2, 1, 0, 0, 0],
        "y_zb": False, "y_down": False, "y_down_streak": 0, "max_h": [0, 2, 1, 0, 0, 0],
        "theme": ["光通信", "短剧影视", None, "光通信", "光通信", "光通信"], "reason": None, "reason_day": None,
    }, index=CODES)
    pos = np.array([0, 3, 4, 5])
    concepts = pd.DataFrame({"code": ["URFI885001", "URFI885002", "URFI885003"], "name": ["CPO概念", "光模块", "短剧"],
                             "kind": "概念", "n": [4, 4, 2], "theme": ["光通信", "光通信", "短剧影视"],
                             "pos": [pos, pos.copy(), np.array([1, 2])]})
    return Context(day=20261013, stocks=st, concepts=concepts,
                   candidates=candidates if candidates is not None else pd.DataFrame(),
                   pos={c: k for k, c in enumerate(CODES)})


def _snap(prices, amounts=None, highs=None) -> pd.DataFrame:
    p = np.asarray(prices, float)
    return pd.DataFrame({"price": p, "amount": np.full(len(p), 5e7) if amounts is None else amounts,
                         "high": p if highs is None else highs, "low": p, "open": p, "up_limit": 11.0,
                         "down_limit": 9.0, "prev_close": 10.0}, index=CODES)


def _t(h, m):
    return datetime(2026, 10, 14, h, m)


def test_auction_signals():
    eng = Engine(_ctx(), load_rules(), CFG)
    seal = {"USHA600002": {"price": 11.0, "bid_left": 1e6, "matched": 1e5, "ask_left": 0}}
    al = eng.auction(_t(9, 25), _snap([10.4, 11.0, 9.3, 10.0, 10.0, 10.0], [3e7, 5e7, 1e6, 1e6, 1e6, 1e6]), seal)
    got = {(a.kind, a.code): a.level for a in al if a.code}
    assert got[("竞价抢筹", "USHA600001")] == "中"          # 3000 万，昨日的 15%，+4%
    assert got[("竞价涨停", "USHA600002")] == "高"          # 昨日 2 板，3 板一字候选
    assert got[("核按钮", "USHA600003")] == "高"            # 昨日涨停，竞价 -7%（≤ -0.6 个涨停）
    assert "竞价封单 1100万" in next(a.detail for a in al if a.kind == "竞价涨停")
    assert eng.stats["auc_limit"] == 1 and eng.up_now[1]   # 竞价涨停视为已封板


def _rules(**sector):
    r = load_rules()
    r["sector"].update(sector)
    return r


def test_open_is_stricter_for_sector_alerts():
    eng = Engine(_ctx(), _rules(), CFG)   # 默认 9:40 前集体异动要 ≥ 5 只
    eng.auction(_t(9, 25), _snap([10.0, 11.0, 10.0, 10.0, 10.0, 10.0], [1e6] * 6))
    a1 = eng.step(_t(9, 31), _snap([10.1, 11.0, 10.0, 10.3, 10.25, 10.22]))
    assert not [a for a in a1 if a.kind == "板块异动"]


def test_sector_drop_alert():
    eng = Engine(_ctx(), _rules(open_strict_until="09:30"), CFG)
    eng.auction(_t(9, 25), _snap([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], [1e6] * 6))
    eng.step(_t(9, 50), _snap([10.0] * 6))
    a = eng.step(_t(9, 53), _snap([9.75, 10.0, 10.0, 9.7, 9.0, 9.72]))   # 4 只下挫 / 跌停（跳水要比拉升多 1 只）
    drop = [x for x in a if x.kind == "板块跳水"]
    assert len(drop) == 1 and "4 只下挫" in drop[0].title and "戊 跌停" in drop[0].detail


def test_continuous_alerts_sector_leader_and_cooldown():
    eng = Engine(_ctx(), _rules(open_strict_until="09:30"), CFG)
    eng.auction(_t(9, 25), _snap([10.0, 11.0, 10.0, 10.0, 10.0, 10.0], [1e6] * 6))
    a1 = eng.step(_t(9, 31), _snap([10.1, 11.0, 10.0, 10.3, 10.25, 10.22]))
    sec = [a for a in a1 if a.kind == "板块异动"]
    assert len(sec) == 1 and "3 只" in sec[0].title and "相关板块：光模块" in sec[0].detail   # 重合板块合并成一条
    assert [a.code for a in a1 if a.kind == "快速拉升"] == ["USHA600004"]                 # +3%（0.3 个涨停）
    assert not [a for a in a1 if a.kind == "封涨停"]                                      # 一字的乙已在竞价封板
    a2 = eng.step(_t(9, 32), _snap([10.1, 11.0, 10.0, 10.3, 11.0, 10.22]))
    up = [a for a in a2 if a.kind == "封涨停"]
    assert [a.code for a in up] == ["USHA600005"] and up[0].level == "高" and "龙头" in up[0].title
    a3 = eng.step(_t(9, 33), _snap([10.1, 10.8, 10.0, 10.3, 10.9, 10.22], highs=[10.1, 11.0, 10, 10.3, 11.0, 10.22]))
    zb = {a.code: a.level for a in a3 if a.kind == "炸板"}
    assert zb == {"USHA600005": "低", "USHA600002": "中"}   # 首板炸板低；3 板（一字开板）中
    assert eng.stats["n_zb"] == 2 and eng.stats["n_up"] == 0
    a4 = eng.step(_t(9, 35), _snap([10.1, 10.8, 9.0, 10.65, 10.9, 10.22]))
    assert not [a for a in a4 if a.kind == "快速拉升" and a.code == "USHA600004"]          # 10 分钟冷却
    assert [a.code for a in a4 if a.kind == "大面"] == ["USHA600003"]                       # 昨日涨停今日跌停
    assert len(eng.sector_tbl) and eng.sector_tbl["name"].iloc[0] in {"CPO概念", "光模块", "短剧"}


def test_first_up_from_limit_stats_is_beijing_time():
    """limit_stats() 的首次涨停时间经 DataFrame 后是 naive pd.Timestamp：不能按 UTC 算（10-15、10-16 实盘晚了 8 小时）。"""
    eng = Engine(_ctx(), _rules(open_strict_until="09:30"), CFG)
    eng.auction(_t(9, 25), _snap([10.0, 11.0, 10.0, 10.0, 10.0, 10.0], [1e6] * 6))
    eng.step(_t(9, 31), _snap([10.1, 11.0, 10.0, 10.3, 10.25, 10.22]))
    fu = pd.DataFrame({"first_up": [datetime(2026, 10, 14, 9, 31, 40)]}, index=["USHA600005"])["first_up"]
    eng.step(_t(9, 32), _snap([10.1, 11.0, 10.0, 10.3, 11.0, 10.22]), dict(fu.items()))
    lim = eng.limit_tbl.set_index("code")
    assert lim.loc["USHA600005", "first_up"] == "09:31:40"


def test_sector_table_shows_last_sector_alert():
    eng = Engine(_ctx(), _rules(open_strict_until="09:30"), CFG)
    eng.auction(_t(9, 25), _snap([10.0, 11.0, 10.0, 10.0, 10.0, 10.0], [1e6] * 6))
    a1 = eng.step(_t(9, 31), _snap([10.1, 11.0, 10.0, 10.3, 10.25, 10.22]))
    fired = {a.sector for a in a1 if a.kind == "板块异动"}
    shown = eng.sector_tbl.set_index("name")["last_alert"]
    assert {n for n, v in shown.items() if v} == fired and set(shown[list(fired)]) == {"09:31"}


def _candidates(n=10):
    first = [9 * 3600 + 1800 + k * 1800 for k in range(n)]   # 09:30 起每半小时一只：越晚越弱
    return pd.DataFrame({
        "key_code": [f"USZA00{1000 + k}" for k in range(n)], "key_name": [f"候选{k}" for k in range(n)],
        "key_t1_date": 20261013, "key_t_date": 99999999, "key_preview": True, "key_level": 1,
        "x_board": "main", "x_limit_pct": 0.1, "x_bt": 1, "x_t1_open_norm": 0.1, "x_t1_dd": 0.3, "x_t1_turnover": 0.1,
        "x_t1_auc_pct": 0.01, "x_t1_first_sec": [float(s) for s in first],
        "x_t1_first_time": [f"{s // 3600:02d}:{s % 3600 // 60:02d}:00" for s in first], "x_t1_reason": "测试",
    })


def test_live_strategy_signals_match_backtest_rules():
    cand = _candidates()
    codes = cand["key_code"].tolist()
    st = pd.DataFrame({"name": cand["key_name"].to_numpy(), "board": "main", "limit_pct": 0.1, "y_close": 10.0,
                       "y_amount": 1e8, "y_amount5": 1e8, "y_up": True, "y_streak": 1, "y_zb": False, "y_down": False,
                       "y_down_streak": 0, "max_h": 1, "theme": None, "reason": None, "reason_day": None}, index=codes)
    ctx = Context(day=20261013, stocks=st, concepts=pd.DataFrame(columns=["code", "name", "kind", "n", "theme", "pos"]),
                  candidates=cand, pos={c: k for k, c in enumerate(codes)})
    eng = Engine(ctx, load_rules(), CFG)
    auc = [10.1, 10.1, 10.15, 10.1, 10.2, 10.1, 10.15, 10.4, 10.5, 10.6]   # 最弱的三只竞价最高
    snap = pd.DataFrame({"price": auc, "amount": 1e7, "up_limit": 11.0, "down_limit": 9.0, "prev_close": 10.0}, index=codes)
    eng.auction(_t(9, 25), snap)
    s = eng.strategy_tbl.set_index("key_code")
    assert s.loc["USZA001009", "x_P_str"] == 1.0 and s.loc["USZA001009", "x_P_auc"] == 0.0
    assert set(s.index[s["sig_A"]]) == {"USZA001008", "USZA001009"}            # 弱半区 + 竞价分位前 20%
    assert set(s.index[s["sig_top"]]) == {"USZA001008", "USZA001009"}          # 每日前 2（按 Δ）
    assert any(a.kind == "一进二信号" and a.level == "高" for a in eng.alerts)
