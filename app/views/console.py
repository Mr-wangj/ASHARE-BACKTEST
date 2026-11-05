"""看板页面：数据控制台 —— 在页面上触发各数据更新任务并实时看日志。

按钮用 subprocess 后台启动（start_new_session 脱离看板进程，关掉看板任务也继续跑），
stdout/stderr 写进 data/logs/console_{任务}.log，pid 写进同名 .pid 文件；
运行中时本页每 5 秒自动刷新日志尾部。任务完成（进程退出）后 pid 文件自动清掉。

盘中保护：工作日 9:10~15:05 禁止启动任务（同花顺账号同时登录会互相挤掉，
且盘中监控正在用这个账号，见使用说明 2.2）。工程自身的锁（daily_update /
live_monitor）仍是最后防线：即便绕过页面限制，重复任务也会自行退出。
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime

import streamlit as st

from ashare_bt.config import DATA_DIR, PROJECT_ROOT

LOG_DIR = DATA_DIR / "logs"
PY = str(PROJECT_ROOT / ".venv" / "bin" / "python")

# key: (标题, 说明, 命令, 预计耗时)
TASKS: dict[str, tuple[str, str, list[str], str]] = {
    "update": ("每日更新（按需）", "自动判断：日线 → 问财明细 → 回测与每日明细 → 舆情，与 15:45 定时任务相同",
               ["scripts/daily_update.py"], "15～30 分钟"),
    "update_force": ("强制完整更新", "不管是否最新，完整重跑一遍（--force）", ["scripts/daily_update.py", "--force"],
                     "15～30 分钟"),
    "news": ("刷新明日题材预判", "不下载新数据，只按本地快讯重算预判模型（改了 config/news_rules.yaml 后用）",
             ["scripts/download_news.py", "--predict-only"], "约 1 分钟"),
    "backtest": ("重跑回测与每日明细", "重算明细、天梯、情绪、回测报告（改了 settings.yaml 或股票池后用）",
                 ["scripts/run_backtest.py"], "约 1 分钟"),
    "concepts": ("刷新同花顺板块成分", "重建板块成分（data/live/concepts.parquet，板块大换血后用）",
                 ["scripts/live_monitor.py", "--build-concepts"], "约 3 分钟"),
    "once": ("盘中监控自检", "取一次实时快照跑一遍引擎（--once，写日志不写预警文件）",
             ["scripts/live_monitor.py", "--once"], "约 1 分钟"),
    "replay": ("回放某交易日（盘中监控）", "用当日分时逐分钟重放；覆盖该日已有记录",
               ["scripts/live_monitor.py", "--replay"], "首次约 7 分钟"),
}


def pid_alive(pid: int) -> bool:
    """任务进程是否存活。Popen 的子进程退出后若父进程不 wait 会变僵尸（defunct），
    os.kill(pid,0) 对僵尸也返回成功 —— 先 waitpid 顺手回收，再用 ps 状态判定（Z = 已退出）。"""
    try:
        os.waitpid(pid, os.WNOHANG)          # 是本进程的子进程：已退出则回收僵尸；运行中不阻塞
    except (ChildProcessError, OSError):
        pass                                 # 不是子进程（如看板重启前的旧任务）：只按存在性判断
    r = subprocess.run(["ps", "-p", str(pid), "-o", "state="], capture_output=True, text=True, check=False)
    state = r.stdout.strip()
    return bool(state) and not state.startswith("Z")


def running_task() -> tuple[str, int, float] | None:
    """当前在跑的控制台任务：(key, pid, 启动时间)。任务结束（含崩溃）时清掉 pid 文件。"""
    for p in LOG_DIR.glob("console_*.pid"):
        key = p.name[8:-4]
        if key not in TASKS:
            p.unlink(missing_ok=True)
            continue
        pid, t0 = p.read_text().split()
        if pid_alive(int(pid)):
            return key, int(pid), float(t0)
        p.unlink(missing_ok=True)
        log = LOG_DIR / f"console_{key}.log"
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== 进程 {pid} 已退出（{datetime.now().strftime('%F %T')}）=====\n")
    return None


def start(key: str, cmd: list[str] | None = None) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = cmd or TASKS[key][2]
    log = open(LOG_DIR / f"console_{key}.log", "a", encoding="utf-8")
    log.write(f"\n===== {datetime.now().strftime('%F %T')} 启动：{' '.join(cmd)} =====\n")
    p = subprocess.Popen([PY, *cmd], cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    (LOG_DIR / f"console_{key}.pid").write_text(f"{p.pid} {time.time()}")
    log.close()


def in_trading_hours() -> bool:
    now = datetime.now()
    return now.weekday() < 5 and (9, 10) <= (now.hour, now.minute) <= (15, 5)


def tail(path, n: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "（还没有日志）"


st.title("数据控制台")
st.caption("按钮在后台启动任务（看板关掉也继续跑），日志在 data/logs/console_任务名.log，本页运行中每 5 秒自动刷新。"
           "任务互斥：一次只允许一个。工作日 9:10～15:05 禁止启动（同花顺账号冲突，盘中监控正在用）；"
           "需要的话等收盘后点，或看 launchd 定时任务（15:45 自动更新）。")

trading = in_trading_hours()
run = running_task()


@st.fragment(run_every=5 if run else None)
def panel() -> None:
    r = running_task()
    if r:
        key, pid, t0 = r
        title, _, _, est = TASKS[key]
        mins = (time.time() - t0) / 60
        st.info(f"**运行中**：{title}（pid {pid}，已 {mins:.0f} 分钟；预计 {est}）。完成后本页自动恢复按钮，"
                "其他页面刷新即可看到新数据。")
        st.code(tail(LOG_DIR / f"console_{key}.log", 40), language=None)
    else:
        if run:                              # 刚从运行中转为结束：整页刷新，恢复任务按钮
            st.rerun(scope="app")
        st.success("当前没有控制台任务在跑。")


panel()

if run:
    st.stop()

st.subheader("数据任务")
cols = st.columns(2)
for i, (k, (title, desc, _, est)) in enumerate(TASKS.items()):
    if k == "replay":
        continue
    with cols[i % 2]:
        c1, c2 = st.columns([3, 1.2])
        c1.markdown(f"**{title}**  \n<span style='opacity:.7;font-size:12.5px'>{desc} · 约 {est}</span>",
                    unsafe_allow_html=True)
        if st.button("启动", key=f"btn_{k}", disabled=trading, use_container_width=True):
            start(k)
            st.rerun()
if trading:
    st.warning("现在是交易时段（9:10～15:05），任务已禁用：盘中运行下载 / 问财类脚本会挤掉同花顺账号、"
               "影响盘中监控。收盘后再来，或用 launchd 定时任务。")

st.subheader("回放某交易日（盘中监控）")
st.caption("用当日分时逐分钟重放（首次回放某天需下载全市场分时，约 7 分钟）。会覆盖该日已有的监控记录；"
           "不要回放今天，也不要在盘中运行。")
rd = st.text_input("日期（YYYYMMDD）", value="", placeholder="如 20261023", key="replay_day")
if st.button("启动回放", disabled=trading or not (rd and rd.isdigit() and len(rd) == 8), key="btn_replay"):
    start("replay", ["scripts/live_monitor.py", "--replay", rd])
    st.rerun()
