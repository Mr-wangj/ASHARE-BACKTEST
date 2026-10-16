"""盘中实时数据（thsdk / 同花顺）。实测速度（2026-10）：

- stocks()：全市场个股快照（问财实时字段，2000 只一批约 0.2 秒）：最新价、涨跌幅、成交额、量比、换手率、涨速、
  开盘 / 最高 / 最低、涨停价 / 跌停价 / 前收盘价（交易所口径，已处理除权）；
- indices()：同花顺 630 个板块指数（概念 885/886、行业 881、地域 882、特色 883，如“昨日涨停表现”）的实时涨跌，1 次调用；
- limit_stats()：首次 / 最终涨停时间、涨停状态（批量）；
- change_events()：同花顺盘口异动（板块异动快讯，如“PCB概念股涨幅扩大，满坤科技20cm涨停”）；
- auction_last()：个股集合竞价最后一笔（9:25 撮合后未匹配的买卖量，即竞价封单），每只约 0.07 秒；
- intraday_bars()：个股分时（回放用；第一根是 9:25 竞价撮合结果），每只约 0.1 秒；
- build_concepts()：板块成分股（同花顺概念约 380 个按名称取成分 + 二级行业约 90 个用个股行业映射反推，约 3 分钟，每周刷新）。
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime

import pandas as pd
import thsdk

from ..config import DATA_DIR, RAW_DIR
from ..data.ths_client import call
from ..data.wencai_limitup import cn_date, extract_field

log = logging.getLogger(__name__)

LIVE_DIR = DATA_DIR / "live"
CONCEPTS_PATH = LIVE_DIR / "concepts.parquet"
STOCK_QUERY = "最新价，涨跌幅，成交额，量比，换手率，涨速，开盘价，最高价，最低价，涨停价，跌停价，前收盘价"
INDEX_QUERY = "最新价，涨跌幅，成交额，涨速"
COLS = {"最新价": "price", "涨跌幅": "pct", "成交额": "amount", "量比": "vol_ratio", "换手率": "turnover", "涨速": "speed",
        "开盘价": "open", "最高价": "high", "最低价": "low", "涨停价": "up_limit", "跌停价": "down_limit",
        "前收盘价": "prev_close"}
PCT_COLS = ("pct", "turnover", "speed")   # 问财返回的是百分数
SPEC_COLS = ["query_group", "query_key", "index_name", "show_name", "timestamp", "index", "unit", "type"]
BATCH = 2000
CONCEPT_PREFIX = {"885": "概念", "886": "概念", "881": "行业"}


def ths_minute(v) -> datetime:
    """同花顺分时的打包时间：(年-1900)<<20 | 月<<16 | 日<<11 | 时<<6 | 分。"""
    v = int(v)
    return datetime(1900 + (v >> 20), (v >> 16) & 0xF, (v >> 11) & 0x1F, (v >> 6) & 0x1F, v & 0x3F)


def epoch(v) -> datetime | None:
    """epoch 秒 → 北京时间（naive）；0 / 空返回 None。"""
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return pd.Timestamp(v, unit="s", tz="Asia/Shanghai").tz_localize(None).to_pydatetime()


class Feed:
    def __init__(self):
        self._spec: dict[str, tuple[date, list[dict]]] = {}

    def _fields(self, query: str) -> list[dict]:
        """问财实时字段定义（按日期缓存：定义里带了查询日期）。"""
        today = date.today()
        hit = self._spec.get(query)
        if hit and hit[0] == today:
            return hit[1]
        spec = call(thsdk.list_wencai_expression_fields, query)
        fields = spec[SPEC_COLS].to_dict("records")
        for f in fields:
            f["unit"] = "" if pd.isna(f["unit"]) else str(f["unit"])
            f["index"] = int(f["index"])
            f["timestamp"] = str(f["timestamp"])
        self._spec[query] = (today, fields)
        return fields

    def _realtime(self, codes: list[str], query: str) -> pd.DataFrame:
        fields = self._fields(query)
        frames = []
        for i in range(0, len(codes), BATCH):
            df = call(thsdk.query_wencai_realtime_fields, securities=codes[i:i + BATCH], fields=fields)
            if df is None or df.empty:
                continue
            vals = pd.DataFrame([{c["name"]: c.get("value") for c in cells if c.get("name") != "code"}
                                 for cells in df["cells"]])
            vals.index = df["full_code"].to_numpy()
            frames.append(vals)
        out = pd.concat(frames) if frames else pd.DataFrame()
        out = out.rename(columns=COLS)
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        for c in PCT_COLS:
            if c in out.columns:
                out[c] = out[c] / 100
        return out[~out.index.duplicated()]

    def stocks(self, codes: list[str]) -> pd.DataFrame:
        return self._realtime(codes, STOCK_QUERY)

    def indices(self, codes: list[str]) -> pd.DataFrame:
        return self._realtime(codes, INDEX_QUERY)

    def limit_stats(self, codes: list[str], batch: int = 500) -> pd.DataFrame:
        """首次 / 最终涨停时间（北京时间）、涨停状态。"""
        metrics = [thsdk.RealtimeStatsMetric.FIRST_LIMIT_UP_TIME, thsdk.RealtimeStatsMetric.FINAL_LIMIT_UP_TIME,
                   thsdk.RealtimeStatsMetric.LIMIT_UP_STATE]
        rows = []
        for i in range(0, len(codes), batch):
            df = call(thsdk.calculate_security_realtime_statistics, securities=codes[i:i + batch], metrics=metrics)
            if df is None or df.empty:
                continue
            for sec, cells in zip(df["security"], df["cells"]):
                v = {c["name"]: c.get("value") for c in cells}
                rows.append({"code": f"{sec.get('market', '')}{sec.get('code', '')}",
                             "first_up": epoch(v.get("首次涨停时间")), "final_up": epoch(v.get("最终涨停时间")),
                             "up_state": v.get("涨停状态")})
        return pd.DataFrame(rows, columns=["code", "first_up", "final_up", "up_state"]).set_index("code")

    def change_events(self, code_map: dict[str, str] | None = None) -> pd.DataFrame:
        """同花顺盘口异动：id、time、block（板块代码:名称）、text、codes（全码）、names。"""
        df = call(thsdk.list_market_change_events)
        cols = ["id", "time", "block", "text", "codes", "names"]
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        rows = []
        for r in df.to_dict("records"):
            pairs = [str(x).split(":", 1) for x in (r.get("unresolved_codes") or []) if ":" in str(x)]
            rows.append({
                "id": str(r.get("third_party_id") or r.get("published_at")),
                "time": pd.Timestamp(r["published_at"]).tz_convert("Asia/Shanghai").tz_localize(None).to_pydatetime(),
                "block": str(r.get("title") or ""), "text": str(r.get("main_title") or ""),
                "codes": [(code_map or {}).get(c, c) for c, _ in pairs], "names": [n for _, n in pairs],
            })
        return pd.DataFrame(rows, columns=cols)

    def wencai_auction(self, day: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        """问财：当日 9:25 集合竞价的成交额、成交价、涨跌幅（小数），index = 6 位代码。

        实时字段接口不支持分时字段（9 点 25 分成交额全为空），问财自然语言可以：“日期 9点25分成交额大于0，……”返回全市场
        （约 5200 只，约 4 秒）。问财每次只返回其中一部分字段、哪部分不固定，所以依次用几种句式，合并取到的结果。"""
        D, tag = cn_date(day), f"{day} 09:25"
        amount, price, pct = {}, {}, {}
        texts = (f"{D}9点25分成交额大于0，{D}9点25分成交额，{D}开盘价",
                 f"{D}9点25分成交额大于0，{D}开盘涨幅",
                 f"{D}9点25分成交额大于0，{D}9点25分成交额，{D}9点25分收盘价")
        for text in texts:
            df = call(thsdk.wencai_nlp, text, limit=8000)
            for target, field, tg in ((amount, "分时成交额", tag), (price, "开盘价", str(day)), (price, "分时收盘价", tag),
                                      (pct, "分时涨跌幅", tag)):
                s = pd.to_numeric(extract_field(df, field, tg), errors="coerce").dropna()
                for k, v in s.items():
                    target.setdefault(k, float(v))
            if len(amount) > 4000 and len(price) + len(pct) > 4000:
                break
        return (pd.Series(amount, dtype="float64"), pd.Series(price, dtype="float64"),
                pd.Series(pct, dtype="float64") / 100)   # 问财涨跌幅为百分数

    def auction_last(self, code: str, day: int) -> dict | None:
        """集合竞价最后一笔：price（撮合价）、matched（匹配量，股）、bid_left / ask_left（未匹配买 / 卖量，股）。"""
        d = datetime.strptime(str(day), "%Y%m%d")
        df = call(thsdk.list_security_call_auction_quotes, retries=2, security=code, phase="opening",
                  date=f"{d:%Y-%m-%d}T09:15:00+08:00")
        if df is None or df.empty:
            return None
        r = df.iloc[-1]
        return {"price": float(r["price"]), "matched": float(r["cur_volume"]), "bid_left": float(r["bid2_vol"]),
                "ask_left": float(r["ask2_vol"]), "time": epoch(r["time"])}

    def intraday_bars(self, code: str, day: int) -> pd.DataFrame | None:
        """分时：time、price、volume / turnover（当日累计）。第一根（09:30）为 9:25 竞价撮合。"""
        df = call(thsdk.list_security_intraday_bars, retries=2, security=code, date=str(day))
        if df is None or df.empty:
            return None
        return pd.DataFrame({"time": [ths_minute(v) for v in df["time"]], "price": df["price"].astype("float64"),
                             "volume": df["volume"].astype("float64"), "turnover": df["turnover"].astype("float64")})

    def index_list(self) -> pd.DataFrame:
        df = call(thsdk.list_market_securities, market="URFI", sort_begin=0, sort_count=5000)
        return df[["full_code", "name"]].reset_index(drop=True)


def industry_members(codes: list[str], batch: int = 1000) -> list[dict]:
    """同花顺二级行业（881）的成分：按名称取不到，改用个股 → 行业映射（批量，1000 只约 13 秒）反推。"""
    groups: dict[tuple[str, str], list[str]] = {}
    for i in range(0, len(codes), batch):
        df = call(thsdk.list_security_industry_mappings, securities=codes[i:i + batch])
        if df is None or df.empty:
            continue
        for sec, ind in zip(df["security"], df["industry"]):
            if isinstance(ind, dict) and isinstance(ind.get("security"), dict):
                key = (f"{ind['security'].get('market', '')}{ind['security'].get('code', '')}", str(ind.get("name", "")))
                groups.setdefault(key, []).append(f"{sec.get('market', '')}{sec.get('code', '')}")
    return [{"code": c, "name": n, "kind": "行业", "members": m, "n": len(m)} for (c, n), m in groups.items()]


def build_concepts(feed: Feed, pause: float = 0.05) -> pd.DataFrame:
    """同花顺概念（885/886）与二级行业（881）的成分股：code（URFI 指数代码）、name、kind、members（个股全码）、n。"""
    idx = feed.index_list()
    idx = idx[idx["full_code"].str[4:7].isin(["885", "886"])]
    rows, t0 = [], time.time()
    for k, r in enumerate(idx.itertuples(), 1):
        try:
            blk = call(thsdk.resolve_block, retries=2, name=r.name)
            mem = call(thsdk.list_block_constituents, retries=2, block=blk, sort_begin=0, sort_count=0,
                       sort_id="55", sort_order="A")
        except Exception as exc:  # noqa: BLE001
            log.warning("板块 %s 成分获取失败：%s", r.name, exc)
            continue
        members = [] if mem is None or mem.empty else [c for c in mem["full_code"] if c[:4] in ("USHA", "USZA", "USTM")]
        rows.append({"code": r.full_code, "name": r.name, "kind": CONCEPT_PREFIX[r.full_code[4:7]],
                     "members": members, "n": len(members)})
        if k % 100 == 0:
            log.info("板块成分 %d/%d，用时 %.0fs", k, len(idx), time.time() - t0)
        time.sleep(pause)
    codes = pd.read_parquet(RAW_DIR / "securities.parquet", columns=["full_code"])["full_code"].tolist()
    rows += industry_members(codes)
    out = pd.DataFrame(rows)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CONCEPTS_PATH, index=False)
    return out
