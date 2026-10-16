"""策略选择只能依赖 key_/x_ 列：打乱或删除 y_ 列不应改变任何选择结果。"""
import numpy as np
import pandas as pd

from ashare_bt.strategy.weak_to_strong import VARIANTS, select

PARAMS = {"p_str_min": 0.5, "p_auc_max": 0.2, "strong_group_max": 0.3, "delta_min": 0.4,
          "auc_norm_lo": 0.2, "auc_norm_hi": 0.7}


def _fake_events(n_days=30, per_day=20, seed=1):
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    ev = pd.DataFrame({
        "key_t_date": np.repeat(np.arange(20250101, 20250101 + n_days), per_day),
        "key_code": [f"C{i}" for i in range(n)],
        "x_P_str": rng.uniform(0, 1, n),
        "x_auc_norm": rng.uniform(-0.3, 1.0, n),
        "x_open_limit": rng.uniform(0, 1, n) > 0.9,
        "x_pool_n": per_day,
    })
    ev["x_P_auc"] = ev.groupby("key_t_date")["x_auc_norm"].rank(ascending=False, pct=True)
    ev["x_delta"] = ev["x_P_str"] - ev["x_P_auc"]
    ev["y_t_close_up"] = rng.uniform(0, 1, n) > 0.7
    ev["y_t2_open"] = rng.uniform(5, 20, n)
    return ev


def test_selection_ignores_outcomes():
    ev = _fake_events()
    shuffled = ev.copy()
    rng = np.random.default_rng(2)
    for c in [c for c in ev.columns if c.startswith("y_")]:
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
    no_y = ev[[c for c in ev.columns if not c.startswith("y_")]]
    for v in VARIANTS:
        a = select(ev, v, PARAMS, min_pool=8, top_n=2)
        b = select(shuffled, v, PARAMS, min_pool=8, top_n=2)
        c = select(no_y, v, PARAMS, min_pool=8, top_n=2)
        assert a.equals(b) and a.equals(c), v


def test_open_limit_never_selected():
    ev = _fake_events()
    for v in VARIANTS:
        idx = select(ev, v, PARAMS, min_pool=8)
        assert not ev.loc[idx, "x_open_limit"].any(), v


def test_top_n_cap():
    ev = _fake_events()
    for v in ["A", "B", "C"]:
        idx = select(ev, v, PARAMS, min_pool=8, top_n=2)
        assert ev.loc[idx].groupby("key_t_date").size().max() <= 2
