"""使用说明：渲染 docs/使用说明.md。

- 左侧按章节（二级标题）浏览，底部有上一章 / 下一章；也可以选「全文」，再用浏览器的页内查找；
- 顶部搜索框按小节（三级标题）检索，显示所有包含关键词的小节；
- 按文件修改时间缓存：文档改完保存后刷新本页即可。
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import streamlit as st

DOC = Path(__file__).resolve().parents[2] / "docs" / "使用说明.md"
ALL = "全文"
TOC = re.compile(r"<!-- toc -->.*?<!-- /toc -->\s*", re.S)   # 文件里的目录：页面左侧已有章节导航，不重复显示


def _split(lines: list[str], prefix: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """按标题前缀（"## " / "### "）切分；代码块里以 # 开头的行（shell 注释）不算标题。

    返回 (第一个标题之前的行, [(标题, 含标题行在内的各行)])。"""
    head: list[str] = []
    parts: list[tuple[str, list[str]]] = []
    in_code = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith(prefix):
            parts.append((line[len(prefix):].strip(), [line]))
        elif parts:
            parts[-1][1].append(line)
        else:
            head.append(line)
    return head, parts


@st.cache_data(show_spinner=False)
def load(mtime: float) -> tuple[str, list[tuple[str, str, list[tuple[str, str]]]]]:
    """返回 (前言, [(章节标题, 章节 markdown, [(小节路径, 小节 markdown)])])；mtime 只用作缓存键。"""
    text = TOC.sub("", DOC.read_text(encoding="utf-8"))
    intro, chapters = _split(text.splitlines(), "## ")
    intro = [ln for ln in intro if not ln.startswith("# ")]   # 一级标题：页面已有 st.title
    out = []
    for title, body in chapters:
        lead, subs = _split(body, "### ")
        sections = [(title, "\n".join(lead))] if "".join(lead[1:]).strip() else []   # 章节标题与第一个小节之间的正文
        sections += [(f"{title} › {t}", "\n".join(b)) for t, b in subs]
        out.append((title, "\n".join(body), sections))
    return "\n".join(intro).strip(), out


st.title("使用说明")
if not DOC.exists():
    st.error(f"未找到文档 {DOC}。")
    st.stop()
mtime = DOC.stat().st_mtime
intro, chapters = load(mtime)
titles = [t for t, _, _ in chapters]
st.caption(f"内容来自 `docs/使用说明.md`（最后修改 {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M}），"
           "编辑保存后刷新本页即可看到新内容。")

q = st.text_input("搜索", placeholder="输入关键词，如 炸板率、竞价抢筹、19:30、settings.yaml", key="guide_q").strip()
options = [*titles, ALL]
if st.session_state.get("guide_sec") not in options:   # 文档改了章节名：回到第一章
    st.session_state.pop("guide_sec", None)


def _go(step: int) -> None:
    st.session_state["guide_sec"] = titles[titles.index(st.session_state["guide_sec"]) + step]


nav, body = st.columns([1, 4], gap="large")
with nav:
    sec = st.radio("章节", options, key="guide_sec", label_visibility="collapsed", disabled=bool(q))
    if q:
        st.caption("清空搜索框后按章节浏览。")
with body:
    if q:
        hits = [(p, md) for _, _, secs in chapters for p, md in secs if q.lower() in md.lower()]
        if not hits:
            st.info(f"没有找到「{q}」。可以换个说法，或清空搜索框按章节浏览。")
        else:
            st.caption(f"「{q}」出现在 {len(hits)} 个小节：")
            for path, md in hits:
                with st.container(border=True):
                    st.markdown(f"**{path}**")
                    st.markdown(md.split("\n", 1)[1] if "\n" in md else "")
    elif sec == ALL:
        st.markdown(intro)
        for _, md, _ in chapters:
            st.markdown(md)
    else:
        i = titles.index(sec)
        if i == 0:
            st.markdown(intro)
        st.markdown(chapters[i][1])
        st.divider()
        prev, nxt = st.columns(2)
        if i > 0:
            prev.button(f"← {titles[i - 1]}", on_click=_go, args=(-1,), key="guide_prev")
        if i < len(titles) - 1:
            nxt.button(f"{titles[i + 1]} →", on_click=_go, args=(1,), key="guide_next")
