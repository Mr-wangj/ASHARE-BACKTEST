"""本地看板入口：uv run streamlit run app/dashboard.py（默认 http://localhost:8501）

页面（侧边栏导航，并列）：
- 一进二明细（默认页）、二进三明细、三进四明细：同一个页面脚本 app/views/daily_detail.py，按板位区分
- 涨停天梯：app/views/ladder.py；跌停天梯：app/views/down_ladder.py（两者共用 app/views/ui_common.py）
- 情绪监测（按时间维度：大盘成交 / 涨跌家数、涨跌停家数、炸板率、涨停 / 跌停高度、晋级、溢价、亏钱效应、情绪温度）：app/views/mood.py
- 舆情监控（明日题材预判、涨停原因、盘后舆情、消息规则）：app/views/monitor.py
- 盘中监控（实时预警：竞价抢筹、板块集体异动、龙头、涨停 / 炸板等）：app/views/live.py
- 策略回测：app/views/backtest.py
- 使用说明（渲染 docs/使用说明.md）：app/views/guide.py
"""
import streamlit as st

from ashare_bt.features.events import DETAIL_LEVELS, level_name

# 侧边栏宽 240px（默认 300）：导航都是短标题，省出宽度给明细表，1512px 屏幕下表格不用横向滚动
st.set_page_config(page_title="A股连板晋级", layout="wide", initial_sidebar_state=240)

ICONS = {1: ":material/looks_one:", 2: ":material/looks_two:", 3: ":material/looks_3:"}
detail_pages = {
    lv: st.Page("views/daily_detail.py", title=f"{level_name(lv)}明细", icon=ICONS.get(lv, ":material/table_view:"),
                url_path=f"detail-{lv}", default=lv == DETAIL_LEVELS[0])
    for lv in DETAIL_LEVELS
}
nav = st.navigation([*detail_pages.values(),
                     st.Page("views/ladder.py", title="涨停天梯", icon=":material/stairs:", url_path="ladder"),
                     st.Page("views/down_ladder.py", title="跌停天梯", icon=":material/trending_down:",
                             url_path="down-ladder"),
                     st.Page("views/mood.py", title="情绪监测", icon=":material/monitor_heart:", url_path="mood"),
                     st.Page("views/monitor.py", title="舆情监控", icon=":material/campaign:", url_path="monitor"),
                     st.Page("views/live.py", title="盘中监控", icon=":material/notifications_active:", url_path="live"),
                     st.Page("views/backtest.py", title="策略回测", icon=":material/monitoring:"),
                     st.Page("views/guide.py", title="使用说明", icon=":material/menu_book:", url_path="guide")],
                    position="sidebar")
# 明细页按当前页面取板位（1 一进二、2 二进三、3 三进四）；每次运行都在页面脚本之前设置
st.session_state["_page_level"] = next((lv for lv, p in detail_pages.items() if p.title == nav.title), DETAIL_LEVELS[0])
nav.run()
