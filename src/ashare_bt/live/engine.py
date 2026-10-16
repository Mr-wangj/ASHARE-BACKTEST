"""盘中信号引擎（纯计算，不做网络与文件读写；实时监控与分时回放共用）。

auction(t, snap)：9:25 竞价结束后调用一次 —— 竞价抢筹、竞价涨停（一字候选，含封单）、昨日涨停股核按钮 / 超预期、
  竞价强势板块、一进二·弱转强实时信号（复用回测的截面排序与选股函数）。
step(t, snap)：连续竞价期间每次轮询调用 —— 封涨停 / 炸板 / 回封、快速拉升、昨日涨停今日跌停（大面）、
  板块集体异动（窗口内多只成分股同时拉升 / 涨停）、板块龙头（空间龙头、先锋、中军、领涨），以及情绪拐点。
快照：index = 个股全码，列 price、amount（当日累计成交额）、high、low、open、up_limit、down_limit、prev_close
（缺失的列按 NaN 处理；回放时由分时与日线计算）。
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..features.events import add_cross_section_ranks
from ..strategy.weak_to_strong import select
from .context import Context

LEVELS = ["低", "中", "高"]
EPS = 0.005            # 价格比较容差（半分钱）
HIST_MINUTES = 20      # 快照历史保留时长
TOP_SECTORS = 30       # 计算龙头的板块数


@dataclass
class Alert:
    time: datetime
    kind: str
    level: str
    title: str
    detail: str = ""
    code: str | None = None
    name: str | None = None
    sector: str | None = None
    key: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time"] = self.time.isoformat(timespec="seconds")
        return d


def bump(level: str, n: int = 1) -> str:
    return LEVELS[max(0, min(len(LEVELS) - 1, LEVELS.index(level) + n))]


def ts(t: datetime) -> float:
    """naive 北京时间 → epoch 秒。

    pd.Timestamp 的 naive 值 .timestamp() 按 UTC 计算，比 datetime.timestamp()（按本地时区）多 8 小时；
    limit_stats() 的首次涨停时间经 DataFrame 后就是 pd.Timestamp（10-15、10-16 实盘的首次涨停时间因此晚了 8 小时），统一转回 datetime。"""
    if isinstance(t, pd.Timestamp):
        t = t.to_pydatetime()
    return t.timestamp()


def hm(sec: float) -> str:
    return "" if not np.isfinite(sec) else datetime.fromtimestamp(sec).strftime("%H:%M:%S")


def money(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{v / 1e8:.2f}亿" if abs(v) >= 1e8 else f"{v / 1e4:.0f}万"


class Engine:
    def __init__(self, ctx: Context, rules: dict, cfg: dict):
        self.ctx, self.rules, self.cfg = ctx, rules, cfg
        st = ctx.stocks
        self.n = len(st)
        self.codes = st.index.to_numpy()
        self.names = st["name"].astype(str).to_numpy()
        self.lim = st["limit_pct"].fillna(0.1).to_numpy(float)
        self.y_close = st["y_close"].to_numpy(float)
        self.y_amount = st["y_amount"].to_numpy(float)
        self.y_up = st["y_up"].to_numpy(bool)
        self.y_streak = st["y_streak"].to_numpy(int)
        self.board = np.where(self.y_up, self.y_streak + 1, 1)          # 今日若涨停是第几板
        self.theme = st["theme"].to_numpy(object)
        self.max_h = st["max_h"].to_numpy(int)
        self.concepts = ctx.concepts
        self.stock_concepts: dict[int, list[int]] = {}   # 行号 → 所属板块（concepts 的行号）
        for ci, p in enumerate(self.concepts["pos"] if len(self.concepts) else []):
            for i in p:
                self.stock_concepts.setdefault(int(i), []).append(ci)
        # 状态
        self.hist: deque[tuple[datetime, np.ndarray]] = deque()
        self.up_now = np.zeros(self.n, bool)
        self.down_now = np.zeros(self.n, bool)
        self.touched = np.zeros(self.n, bool)
        self.broke_once = np.zeros(self.n, bool)
        self.first_up = np.full(self.n, np.nan)    # epoch 秒
        self.up_time = np.full(self.n, np.nan)     # 最近一次封板时间
        self.down_time = np.full(self.n, np.nan)   # 最近一次跌停时间
        self.big_loss = np.zeros(self.n, bool)
        self.last_alert: dict[str, datetime] = {}
        self.sector_last: dict[str, datetime] = {}
        self.flags: set[str] = set()
        self.leaders: dict[int, str] = {}          # 行号 → 所在强势板块（空间龙头）
        self.hot_pos = np.zeros(self.n, bool)      # 当前最强几个板块的成分股
        self.alerts: list[Alert] = []
        self.auction_tbl = pd.DataFrame()
        self.strategy_tbl = pd.DataFrame()
        self.sector_tbl = pd.DataFrame()
        self.limit_tbl = pd.DataFrame()
        self.stats: dict = {}
        self.t_auction: datetime | None = None
        self.last_t: datetime | None = None

    # ------------------------------------------------------------------ 工具
    def _arrays(self, snap: pd.DataFrame) -> dict[str, np.ndarray]:
        s = snap.reindex(self.codes)
        col = lambda c: s[c].to_numpy(float) if c in s.columns else np.full(self.n, np.nan)  # noqa: E731
        a = {c: col(c) for c in ("price", "amount", "high", "low", "open", "up_limit", "down_limit", "prev_close")}
        pc = np.where(np.isfinite(a["prev_close"]) & (a["prev_close"] > 0), a["prev_close"], self.y_close)
        price = a["price"]
        valid = np.isfinite(price) & (price > 0)
        a["valid"] = valid
        a["pct"] = np.where(valid, price / pc - 1, np.nan)
        a["pct_norm"] = a["pct"] / self.lim
        up, dn, hi = a["up_limit"], a["down_limit"], a["high"]
        a["is_up"] = valid & np.isfinite(up) & (np.abs(price - up) < EPS)
        a["is_down"] = valid & np.isfinite(dn) & (np.abs(price - dn) < EPS)
        a["touch_up"] = a["is_up"] | (np.isfinite(hi) & np.isfinite(up) & (np.abs(hi - up) < EPS))
        return a

    def _cool(self, key: str, t: datetime, minutes: float | None = None) -> bool:
        """冷却期内的同一预警不再发出；可以发出时记录时间并返回 True。"""
        cd = timedelta(minutes=self.rules["cooldown_minutes"] if minutes is None else minutes)
        last = self.last_alert.get(key)
        if last is not None and t - last < cd:
            return False
        self.last_alert[key] = t
        return True

    def _base(self, t: datetime, minutes: float) -> np.ndarray | None:
        """t 之前 minutes 分钟的价格；历史不足一个窗口时用最早的快照（通常是 9:25 竞价价）。"""
        t0 = t - timedelta(minutes=minutes)
        base = None
        for tt, p in self.hist:
            if tt <= t0:
                base = p
            else:
                break
        if base is None and self.hist:
            base = self.hist[0][1]
        return base

    def _add(self, out: list[Alert], a: Alert) -> None:
        out.append(a)
        self.alerts.append(a)

    def _stock_alert(self, t, kind, level, i, title, detail="", key=None) -> Alert:
        th = self.theme[i] if isinstance(self.theme[i], str) else None
        return Alert(t, kind, level, title, detail, code=str(self.codes[i]), name=str(self.names[i]), sector=th,
                     key=key or f"{kind}:{self.codes[i]}")

    def _label(self, i: int) -> str:
        th = self.theme[i] if isinstance(self.theme[i], str) else ""
        return f"{self.names[i]}" + (f"［{th}］" if th else "")

    # ------------------------------------------------------------------ 9:25 竞价
    def auction(self, t: datetime, snap: pd.DataFrame, seal: dict[str, dict] | None = None) -> list[Alert]:
        r, out = self.rules["auction"], []
        a = self._arrays(snap)
        valid, pct, pn, amt = a["valid"], a["pct"], a["pct_norm"], a["amount"]
        ratio = np.where(self.y_amount > 0, amt / self.y_amount, np.nan)
        at_up, at_dn = a["is_up"], a["is_down"]
        g = r["grab"]
        grab = valid & (amt >= g["min_amount"]) & (ratio >= g["min_ratio"]) & (pn >= g["min_pct_norm"]) & ~at_up
        nuke = valid & self.y_up & (pn <= r["yzt_nuke_norm"])
        beyond = valid & self.y_up & (pn >= r["yzt_beyond_norm"]) & ~at_up
        seal = seal or {}
        seal_amt = np.array([seal[c]["bid_left"] * seal[c]["price"] if c in seal else np.nan for c in self.codes])

        # 竞价抢筹：按 竞价额 ÷ 昨日成交额 排序，推送前 N 只
        idx = np.flatnonzero(grab)
        idx = idx[np.argsort(-ratio[idx])]
        for i in idx[: g["max_alerts"]]:
            lv = "高" if ratio[i] >= g["high_ratio"] and amt[i] >= g["high_amount"] else "中"
            self._add(out, self._stock_alert(
                t, "竞价抢筹", lv, i, f"{self._label(i)} 竞价抢筹 {pct[i]:+.1%}",
                f"竞价额 {money(amt[i])}，为昨日全天成交额的 {ratio[i]:.0%}" + ("；昨日涨停" if self.y_up[i] else "")))
        for i in np.flatnonzero(valid & at_up):
            lv = "高" if self.y_up[i] else r["limit_up_level"]
            what = f"{self.board[i]}板一字候选" if self.y_up[i] else "首板一字候选"
            self._add(out, self._stock_alert(
                t, "竞价涨停", lv, i, f"{self._label(i)} 竞价涨停（{what}）",
                f"竞价额 {money(amt[i])}" + (f"，竞价封单 {money(seal_amt[i])}" if np.isfinite(seal_amt[i]) else "")))
        for i in np.flatnonzero(nuke):
            lv = "高" if pn[i] <= 2 * r["yzt_nuke_norm"] else "中"
            self._add(out, self._stock_alert(
                t, "核按钮", lv, i, f"{self._label(i)} 昨日{self.y_streak[i]}板，竞价 {pct[i]:+.1%}",
                f"竞价额 {money(amt[i])}" + ("，竞价跌停" if at_dn[i] else "")))
        for i in np.flatnonzero(beyond):
            self._add(out, self._stock_alert(
                t, "竞价超预期", "中", i, f"{self._label(i)} 昨日{self.y_streak[i]}板，竞价 {pct[i]:+.1%}",
                f"竞价额 {money(amt[i])}，为昨日的 {ratio[i]:.0%}"))

        # 竞价强势板块
        strong = valid & (pn >= 0.3)
        rows = []
        for c in self.concepts.itertuples():
            p = c.pos
            k = int(strong[p].sum())
            if k >= r["sector_min_members"]:
                top = p[np.argsort(-np.nan_to_num(pct[p], nan=-9))][:3]
                rows.append((k, float(np.nanmean(pct[p])), c.name, c.theme, top, int(at_up[p].sum())))
        rows.sort(key=lambda x: (-x[5], -x[0], -x[1]))   # 竞价涨停只数优先（小板块的一字潮），再看高开只数
        for k, avg, name, theme, top, n_lim in rows[: r["sector_alerts"]]:
            self._add(out, Alert(t, "竞价强势板块", "中", f"{name} 竞价强势：{k} 只高开 ≥ 0.3 个涨停",
                                 "、".join(f"{self.names[i]} {pct[i]:+.1%}" for i in top)
                                 + f"；板块均 {avg:+.1%}" + (f"，竞价涨停 {n_lim} 只" if n_lim else ""),
                                 sector=name, key=f"竞价板块:{name}"))

        keep = valid & (amt > 0)
        self.auction_tbl = pd.DataFrame({
            "code": self.codes[keep], "name": self.names[keep], "pct": pct[keep], "amount": amt[keep],
            "ratio": ratio[keep], "at_limit": at_up[keep], "seal_amt": seal_amt[keep], "y_up": self.y_up[keep],
            "y_streak": self.y_streak[keep], "theme": self.theme[keep], "grab": grab[keep], "nuke": nuke[keep],
            "beyond": beyond[keep],
        })
        yz = valid & self.y_up
        self.stats.update({
            "auction_time": t.isoformat(timespec="seconds"), "auc_limit": int((valid & at_up).sum()),
            "auc_down": int((valid & at_dn).sum()), "auc_up5": int((valid & (pct >= 0.05)).sum()),
            "auc_dn5": int((valid & (pct <= -0.05)).sum()), "auc_grab": int(grab.sum()),
            "auc_yzt": float(np.nanmean(pct[yz])) if yz.any() else None, "auc_yzt_n": int(yz.sum()),
        })
        self.strategy_tbl = self.strategy(t, a)
        s = self.strategy_tbl
        if len(s):
            cnt = {v: int(s[f"sig_{v}"].sum()) for v in ("A", "B", "C")}
            picks = s[s["sig_top"]].sort_values("x_delta", ascending=False)
            if any(cnt.values()):
                self._add(out, Alert(
                    t, "一进二信号", "高" if len(picks) else "中",
                    "一进二·弱转强：" + "、".join(f"口径{v} {n} 只" for v, n in cnt.items() if n),
                    "每日前2：" + ("、".join(f"{r.key_name}（T-1序号 {int(r.x_t1_rank)}/{int(r.x_pool_n)}，竞价 "
                                            f"{r.x_auc_pct:+.1%}）" for r in picks.itertuples()) or "无"),
                    key="一进二信号"))
        # 竞价价作为连续竞价的第一个基准；竞价涨停的视为已封板（开盘后打开即炸板）
        self.hist.clear()
        self.hist.append((t, np.where(valid, a["price"], np.nan)))
        self.up_now = valid & at_up
        self.touched |= self.up_now
        self.first_up[self.up_now] = ts(t)
        self.up_time[self.up_now] = ts(t)
        self.t_auction = t
        return out

    def strategy(self, t: datetime, a: dict) -> pd.DataFrame:
        """一进二·弱转强：T-1 首板候选 + 当日竞价 → 截面排序（T-1 强度分位、竞价分位、Δ）→ 口径 A/B/C。"""
        ev = self.ctx.candidates
        if ev is None or ev.empty:
            return pd.DataFrame()
        ev = ev.copy()
        ii = ev["key_code"].map(self.ctx.pos)
        ev = ev[ii.notna()].copy()
        ii = ii[ii.notna()].astype(int).to_numpy()
        ev["x_auc_pct"] = a["pct"][ii]
        ev["x_auc_norm"] = ev["x_auc_pct"] / ev["x_limit_pct"]
        ev["x_open_limit"] = a["is_up"][ii]
        ev["x_t_auc_amt"] = a["amount"][ii]
        ev = ev[np.isfinite(ev["x_auc_pct"].to_numpy(float))].copy()
        if ev.empty:
            return ev
        ev["key_t_date"] = int(t.strftime("%Y%m%d"))
        ev["key_preview"] = False
        add_cross_section_ranks(ev, self.cfg["strategy"]["strength_mode"])
        sp, mp = self.cfg["strategy"], self.cfg["universe"]["min_pool"]
        for v in ("base3", "A", "B", "C"):
            ev[f"sig_{v}"] = ev.index.isin(select(ev, v, sp, mp))
        top: set = set()
        for v in ("A", "B", "C"):
            top |= set(select(ev, v, sp, mp, top_n=sp["top_n"]))
        ev["sig_top"] = ev.index.isin(top)
        return ev.sort_values("x_t1_rank").reset_index(drop=True)

    # ------------------------------------------------------------------ 连续竞价
    def step(self, t: datetime, snap: pd.DataFrame, first_up: dict[str, datetime] | None = None,
             index_pct: dict[str, float] | None = None) -> list[Alert]:
        with warnings.catch_warnings():   # 板块成分全部停牌 / 无价格时 nanmean 会告警，结果为 NaN 即可
            warnings.simplefilter("ignore", RuntimeWarning)
            return self._step(t, snap, first_up, index_pct)

    def _step(self, t: datetime, snap: pd.DataFrame, first_up: dict[str, datetime] | None,
              index_pct: dict[str, float] | None) -> list[Alert]:
        rs, rsec, out = self.rules["stock"], self.rules["sector"], []
        a = self._arrays(snap)
        valid, price, pct, amt = a["valid"], a["price"], a["pct"], a["amount"]
        is_up, is_down = a["is_up"], a["is_down"]
        base = self._base(t, rs["surge_window"])
        base_s = self._base(t, rsec["window"])
        self.hist.append((t, np.where(valid, price, np.nan)))
        while self.hist and t - self.hist[0][0] > timedelta(minutes=HIST_MINUTES):
            self.hist.popleft()
        rise = price / base - 1 if base is not None else np.full(self.n, np.nan)
        rise_s = price / base_s - 1 if base_s is not None else np.full(self.n, np.nan)
        rise_norm, rise_s_norm = rise / self.lim, rise_s / self.lim

        # 涨停状态变化
        new_up = is_up & ~self.up_now
        broke = self.up_now & ~is_up & valid
        now = ts(t)
        if first_up:
            for c, ft in first_up.items():
                i = self.ctx.pos.get(c)
                if i is not None and ft is not None and not np.isfinite(self.first_up[i]):
                    self.first_up[i] = ts(ft)
        self.first_up[new_up & ~np.isfinite(self.first_up)] = now
        self.up_time[new_up] = now
        self.touched |= a["touch_up"]
        resealed = new_up & self.broke_once
        self.broke_once |= broke
        self.up_now = np.where(valid, is_up, self.up_now)
        self.down_time[is_down & ~self.down_now] = now
        self.down_now = np.where(valid, is_down, self.down_now)

        # 板块：3 分钟内拉升 / 新封板（跳水 / 新跌停）的成分股；5 分钟整体涨跌与广度
        win_s = rsec["window"] * 60
        recent_up = np.isfinite(self.up_time) & (self.up_time >= now - win_s) & self.up_now
        recent_dn = np.isfinite(self.down_time) & (self.down_time >= now - win_s) & self.down_now
        mover = valid & ((rise_s_norm >= rsec["member_norm"]) | recent_up)
        faller = valid & ((rise_s_norm <= -rsec["member_norm"]) | recent_dn)
        base5 = self._base(t, rsec["index_window"])
        rise5 = price / base5 - 1 if base5 is not None else np.full(self.n, np.nan)
        up5, dn5 = valid & (rise5 / self.lim >= 0.1), valid & (rise5 / self.lim <= -0.1)
        sec_rows = []
        for c in self.concepts.itertuples():
            p = c.pos
            sec_rows.append((c.Index, int(self.up_now[p].sum()), float(np.nanmean(pct[p])), int(mover[p].sum()),
                             int(faller[p].sum()), float(np.nanmean(rise_s[p])), float(np.nanmean(rise5[p])),
                             float(up5[p].mean()), float(dn5[p].mean()), int((valid[p] & (pct[p] > 0.03)).sum()),
                             int(self.down_now[p].sum()), len(p)))
        st = pd.DataFrame(sec_rows, columns=["ci", "n_up", "avg_pct", "n_mov", "n_fall", "avg_rise", "avg5", "br_up5",
                                             "br_dn5", "n_strong", "n_down", "n"])
        if len(st):
            st = st.sort_values(["n_up", "avg_pct"], ascending=False).reset_index(drop=True)
        top = st.head(TOP_SECTORS)
        self.leaders, self.hot_pos = {}, np.zeros(self.n, bool)
        lead_cols, opts = [], {}
        for k, r in enumerate(top.itertuples()):
            c = self.concepts.iloc[r.ci]
            ld = self._leaders(c["pos"], pct, amt)
            lead_cols.append(ld)
            if ld["space"] is not None and r.n_up > 0:
                i = ld["space"]
                opts.setdefault(i, []).append((k, c["name"], isinstance(self.theme[i], str) and c["theme"] == self.theme[i]))
            if k < 5 and r.n_up > 0:
                self.hot_pos[c["pos"]] = True
        # 是前 leader_top 强板块的空间龙头才算龙头；一只股票属于很多概念，标注时优先选与它涨停原因同题材、
        # 且它也是空间龙头的板块（如慈文传媒标“文化传媒概念”，而不是“旅游概念”）
        for i, o in opts.items():
            if min(k for k, _, _ in o) >= rsec["leader_top"]:
                continue
            label = next((name for _, name, m in o if m), None)
            if label is None and isinstance(self.theme[i], str):
                for ci in self.stock_concepts.get(i, []):
                    c = self.concepts.iloc[ci]
                    if c["theme"] == self.theme[i] and self._leaders(c["pos"], pct, amt)["space"] == i:
                        label = c["name"]
                        break
            self.leaders[i] = label or o[0][1]

        # 个股预警
        for i in np.flatnonzero(new_up):
            b = self.board[i]
            lv = "高" if b >= 3 else ("中" if b == 2 else rs["limit_up_level"])
            if self.hot_pos[i]:
                lv = bump(lv)
            lead = self.leaders.get(i)
            if lead:
                lv = "高"
            kind = "回封" if resealed[i] else "封涨停"
            if kind == "回封":
                lv = "中" if b >= 2 or lead else "低"
            if self._cool(f"{kind}:{self.codes[i]}", t, 5):
                self._add(out, self._stock_alert(
                    t, kind, lv, i, f"{self._label(i)} {kind}（{'首板' if b == 1 else f'{b}板'}）"
                    + (f" · {lead}龙头" if lead else ""),
                    f"首次涨停 {hm(self.first_up[i])}，成交额 {money(amt[i])}"))
        for i in np.flatnonzero(broke):
            b = self.board[i]
            lead = self.leaders.get(i)
            lv = "高" if lead else (rs["break_level"] if b >= 2 else "低")
            if self._cool(f"炸板:{self.codes[i]}", t):
                self._add(out, self._stock_alert(
                    t, "炸板", lv, i, f"{self._label(i)} 炸板（{'首板' if b == 1 else f'{b}板'}）"
                    + (f" · {lead}龙头" if lead else ""), f"现 {pct[i]:+.1%}，首次涨停 {hm(self.first_up[i])}"))
        surge = valid & ~is_up & (rise_norm >= rs["surge_norm"]) & (amt >= rs["surge_min_amount"])
        for i in np.flatnonzero(surge):
            lv = rs["surge_level"]
            if self.y_up[i] or self.hot_pos[i]:
                lv = bump(lv)
            if self._cool(f"快速拉升:{self.codes[i]}", t):
                self._add(out, self._stock_alert(
                    t, "快速拉升", lv, i, f"{self._label(i)} {rs['surge_window']} 分钟拉升 {rise[i]:+.1%}，现 {pct[i]:+.1%}",
                    f"成交额 {money(amt[i])}" + (f"；昨日{self.y_streak[i]}板" if self.y_up[i] else "")))
        for i in np.flatnonzero(valid & is_down & self.y_up & ~self.big_loss):
            self.big_loss[i] = True
            self._add(out, self._stock_alert(t, "大面", self.rules["stock"]["big_loss_level"], i,
                                             f"{self._label(i)} 昨日{self.y_streak[i]}板，今日跌停", ""))

        # 板块集体异动 / 跳水
        if len(st):
            self._sector_alerts(out, t, st, pct, amt, rise_s, rise5, mover, faller, up5, dn5)

        # 情绪：昨日涨停股今日平均表现
        yz = valid & self.y_up
        yzt = float(np.nanmean(pct[yz])) if yz.any() else float("nan")
        sen = self.rules["sentiment"]
        if np.isfinite(yzt) and yzt <= sen["yzt_drop"] and "yzt_drop" not in self.flags:
            self.flags.add("yzt_drop")
            self._add(out, Alert(t, "情绪", "中", f"亏钱效应扩散：昨日涨停股平均 {yzt:+.1%}",
                                 f"昨日涨停 {int(yz.sum())} 只，今日跌停 {int((yz & is_down).sum())} 只", key="情绪:drop"))
        if np.isfinite(yzt) and yzt >= sen["yzt_rise"] and "yzt_rise" not in self.flags:
            self.flags.add("yzt_rise")
            self._add(out, Alert(t, "情绪", "中", f"赚钱效应：昨日涨停股平均 {yzt:+.1%}",
                                 f"昨日涨停 {int(yz.sum())} 只，今日继续涨停 {int((yz & self.up_now).sum())} 只",
                                 key="情绪:rise"))

        # 页面用的表与统计
        self._tables(t, a, st, top, lead_cols, index_pct)
        zb = self.touched & ~self.up_now
        self.stats.update({
            "time": t.isoformat(timespec="seconds"), "n_up": int(self.up_now.sum()), "n_touch": int(self.touched.sum()),
            "n_zb": int(zb.sum()), "n_down": int(self.down_now.sum()),
            "zb_rate": float(zb.sum() / self.touched.sum()) if self.touched.any() else None,
            "yzt_avg": yzt if np.isfinite(yzt) else None, "yzt_up": int((yz & self.up_now).sum()),
            "n_red": int((valid & (pct > 0)).sum()), "n_green": int((valid & (pct < 0)).sum()),
            "median_pct": float(np.nanmedian(pct[valid])) if valid.any() else None,
        })
        self.last_t = t
        return out

    def _last_move(self, code: str) -> str:
        """板块表「最近集体异动」：最近一次板块异动 / 板块跳水的时间（sector_last 的键是“类型:板块代码”）。"""
        up, dn = self.sector_last.get(f"板块异动:{code}"), self.sector_last.get(f"板块跳水:{code}")
        if up is None and dn is None:
            return ""
        if dn is None or (up is not None and up >= dn):
            return up.strftime("%H:%M")
        return f"{dn:%H:%M} 跳水"

    def _sector_alerts(self, out, t, st, pct, amt, rise_s, rise5, mover, faller, up5, dn5) -> None:
        """两种触发（向上 = 板块异动，向下 = 板块跳水，规则对称）：
        - 集体：window 分钟内 ≥ min_members 只成分股拉升 ≥ member_norm 或新封板，且全部成分股平均涨 ≥ min_avg_rise；
        - 整体：index_window 分钟成分股平均涨 ≥ index_rise，且其中涨超 0.1 个涨停的占比 ≥ index_breadth（成分 ≥ index_min_size）。
        同一轮里成分股高度重合的板块合并成一条；同一板块冷却 cooldown_minutes；同一题材 theme_cooldown 分钟内的第二条降为低级。"""
        rsec = self.rules["sector"]
        early = t.strftime("%H:%M") < str(rsec.get("open_strict_until", "09:40"))   # 开盘初段波动大：门槛提高
        fd = float(rsec.get("down_factor", 1.5))
        cands = []
        for r in st.itertuples():
            for d in (1, -1):
                if d < 0 and not rsec.get("down", True):
                    continue
                n_m, avg, a5, br = ((r.n_mov, r.avg_rise, r.avg5, r.br_up5) if d > 0
                                    else (r.n_fall, -r.avg_rise, -r.avg5, r.br_dn5))
                min_m = rsec["high_members"] if early else rsec["min_members"]
                k_idx = (2.0 if early else 1.0) * (fd if d < 0 else 1.0)
                if d < 0:
                    min_m += 1
                cluster = n_m >= min_m and avg >= rsec["min_avg_rise"] * (fd if d < 0 else 1.0)
                broad = (r.n >= rsec["index_min_size"] and a5 >= rsec["index_rise"] * k_idx
                         and br >= rsec["index_breadth"] * (fd if d < 0 else 1.0))
                if cluster or broad:
                    cands.append((d, r, cluster, n_m, a5, br))
        cands.sort(key=lambda x: (-x[3], -x[4]))
        covered: dict[int, set[int]] = {1: set(), -1: set()}
        fired: list[tuple[int, Alert, set[int]]] = []
        for d, r, cluster, n_m, a5, br in cands:
            c = self.concepts.iloc[r.ci]
            p = c["pos"]
            ms = set(p[(mover if d > 0 else faller)[p] | (up5 if d > 0 else dn5)[p]].tolist())
            hit = next((f for f in fired if f[0] == d and ms and len(ms & f[2]) >= rsec["overlap"] * len(ms)), None)
            if hit is not None:
                hit[1].detail += ("、" if "相关板块" in hit[1].detail else "；相关板块：") + c["name"]
                continue
            if ms and len(ms & covered[d]) >= rsec["overlap"] * len(ms):
                continue
            kind = "板块异动" if d > 0 else "板块跳水"
            covered[d] |= ms
            if not self._cool(f"{kind}:{c['code']}", t, rsec["cooldown_minutes"]):
                continue
            skey = f"{kind}:{c['code']}"
            fresh = skey not in self.sector_last or t - self.sector_last[skey] > timedelta(minutes=60)
            self.sector_last[skey] = t
            th = c["theme"] if isinstance(c["theme"], str) else None
            if d > 0:
                lv = "高" if n_m >= rsec["high_members"] and not early else "中"
            else:
                lv = "中" if n_m >= rsec["high_members"] or r.n_down >= 2 else "低"
            dup = th is not None and not self._cool(f"板块题材:{th}:{d}", t, rsec["theme_cooldown"])
            if dup:
                lv = "低"
            ref = rise_s if cluster else rise5
            order = sorted(ms, key=lambda i: -d * (ref[i] if np.isfinite(ref[i]) else -9 * d))
            state = (lambda i: " 涨停" if self.up_now[i] else f" {ref[i]:+.1%}") if d > 0 else \
                    (lambda i: " 跌停" if self.down_now[i] else f" {ref[i]:+.1%}")
            names = "、".join(f"{self.names[i]}{state(i)}" for i in order[:5])
            label = c["name"] + (f"（{th}）" if th and th != c["name"] else "")
            if cluster:
                what = (f"集体异动：{rsec['window']} 分钟内 {n_m} 只拉升 / 封板" if d > 0
                        else f"集体跳水：{rsec['window']} 分钟内 {n_m} 只下挫 / 跌停")
            else:
                what = (f"板块拉升：{rsec['index_window']} 分钟成分股均涨 {a5:+.1%}，{br:.0%} 涨超 0.1 个涨停" if d > 0
                        else f"板块走弱：{rsec['index_window']} 分钟成分股均跌 {-a5:.1%}，{br:.0%} 跌超 0.1 个涨停")
            ld = self._leaders(p, pct, amt)
            detail = (f"{names}；板块均 {r.avg_pct:+.1%}，" + (f"涨停 {r.n_up} 只" if d > 0 else f"跌停 {r.n_down} 只")
                      + (f"；龙头 {self.names[ld['space']]}（{self.board[ld['space']]}板）"
                         if d > 0 and ld["space"] is not None else "")
                      + (f"（同题材「{th}」{rsec['theme_cooldown']} 分钟内已提示）" if dup else ""))
            al = Alert(t, kind, lv, f"{label} {what}" + ("（新启动）" if fresh and d > 0 else ""), detail,
                       sector=c["name"], key=skey)
            self._add(out, al)
            fired.append((d, al, ms))

    def _leaders(self, p: np.ndarray, pct: np.ndarray, amt: np.ndarray) -> dict:
        """空间龙头（封板中、板数最高，并列取先封板的）、先锋（最先涨停）、中军（涨 ≥ 3% 中成交额最大）、领涨。"""
        sealed = p[self.up_now[p]]
        space = None
        if len(sealed):
            order = np.lexsort((np.nan_to_num(self.first_up[sealed], nan=9e18), -self.board[sealed]))
            space = int(sealed[order[0]])
        ups = p[np.isfinite(self.first_up[p])]
        pioneer = int(ups[np.argmin(self.first_up[ups])]) if len(ups) else None
        big = p[np.nan_to_num(pct[p], nan=-9) >= 0.03]
        zhongjun = int(big[np.argmax(np.nan_to_num(amt[big]))]) if len(big) else None
        valid_p = p[np.isfinite(pct[p])]
        top = int(valid_p[np.argmax(pct[valid_p])]) if len(valid_p) else None
        return {"space": space, "pioneer": pioneer, "zhongjun": zhongjun, "top": top}

    def _tables(self, t, a, st, top, lead_cols, index_pct) -> None:
        pct, amt = a["pct"], a["amount"]
        nm = lambda i: "" if i is None else str(self.names[i])  # noqa: E731
        rows = []
        for r, ld in zip(top.itertuples(), lead_cols):
            c = self.concepts.iloc[r.ci]
            rows.append({
                "code": c["code"], "name": c["name"], "theme": c["theme"], "kind": c["kind"], "n": int(c["n"]),
                "avg_pct": r.avg_pct, "idx_pct": (index_pct or {}).get(c["code"]), "n_up": r.n_up, "n_strong": r.n_strong,
                "n_mov": r.n_mov, "avg_rise": r.avg_rise,
                "space": nm(ld["space"]) + (f"（{self.board[ld['space']]}板）" if ld["space"] is not None else ""),
                "pioneer": nm(ld["pioneer"]) + (f" {hm(self.first_up[ld['pioneer']])[:5]}" if ld["pioneer"] is not None else ""),
                "zhongjun": nm(ld["zhongjun"]) + (f" {money(amt[ld['zhongjun']])}" if ld["zhongjun"] is not None else ""),
                "top": nm(ld["top"]) + (f" {pct[ld['top']]:+.1%}" if ld["top"] is not None else ""),
                "last_alert": self._last_move(c["code"]),
            })
        self.sector_tbl = pd.DataFrame(rows)
        k = np.flatnonzero(self.touched)
        self.limit_tbl = pd.DataFrame({
            "code": self.codes[k], "name": self.names[k], "board": self.board[k],
            "first_up": [hm(x) for x in self.first_up[k]], "first_sec": self.first_up[k],
            "sealed": self.up_now[k], "pct": pct[k], "amount": amt[k], "theme": self.theme[k],
            "leader_of": [self.leaders.get(int(i), "") for i in k], "y_streak": self.y_streak[k],
        }).sort_values(["sealed", "board", "first_sec"], ascending=[False, False, True])
