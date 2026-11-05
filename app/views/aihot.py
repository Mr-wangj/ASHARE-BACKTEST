"""看板页面：AIHOT —— 中文 AI 资讯（aihot.news）。

数据来自 AIHOT 的公开 agent 接口（只读 GET，返回整理好的中文 Markdown）：
今日 / 最近一周 / 当前热点 / AIHOT 日报 / 关键词搜索。查询结果缓存 15 分钟。
"""
from __future__ import annotations

import gzip
import urllib.parse
import urllib.request

import streamlit as st

BASE = "https://aihot.news/api/v1/agent"
UA = "aihot-dashboard/1.0 (+https://aihot.news/aihot-skill/)"
MODES = {"latest": "今日 AI", "week": "最近一周", "hot": "当前热点", "daily": "AI 日报", "search": "搜索"}
URLS = {"latest": f"{BASE}/latest", "week": f"{BASE}/latest?window=7d",
        "hot": f"{BASE}/hot", "daily": f"{BASE}/daily"}


@st.cache_data(ttl=900, show_spinner="查询 AIHOT……")
def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors="replace")


st.title("AIHOT · AI 资讯")
st.caption("中文 AI 资讯聚合（aihot.news）：今日动态、热点、日报与搜索；内容缓存 15 分钟。"
           "适合看盘之余扫一眼 AI 圈动静——对 A 股的 AI 应用、算力、传媒（短剧）等题材也是消息面参考。")

mode = st.segmented_control("内容", list(MODES), format_func=MODES.get, default="latest", required=True, key="aihot_mode")
url = URLS.get(mode)
if mode == "search":
    q = st.text_input("关键词（公司 / 产品 / 模型 / 人物 / 话题）", key="aihot_q", placeholder="如 OpenAI、GPT、Claude")
    if q:
        url = f"{BASE}/search?q={urllib.parse.quote(q)}"
    else:
        st.info("输入关键词后按回车查询。")
        st.stop()

if url:
    try:
        md = fetch(url)
        st.markdown(md)
    except Exception as exc:      # noqa: BLE001
        st.warning(f"AIHOT 查询失败（{str(exc)[:100]}）。稍后重试，或直接访问 https://aihot.news/ 。")
