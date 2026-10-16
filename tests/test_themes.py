from collections import Counter

import pandas as pd

from ashare_bt.themes import (
    PALETTE, Group, ThemeMap, assign_colors, categories, category_counts, load_theme_map, main_category, split_tags,
    theme_table,
)


def _tm():
    return ThemeMap([
        Group("短剧影视", "题材", ("短剧", "漫剧", "影视", "院线")),
        Group("网络安全", "题材", ("网络安全", "人工智能安全", "信息安全")),
        Group("算力", "题材", ("算力", "智算", "数据中心")),
        Group("光伏", "题材", ("光伏", "钙钛矿")),
        Group("锂电", "题材", ("固态电池", "电池")),
        Group("业绩", "事件", ("季报", "业绩")),
        Group("国资国企", "事件", ("国资", "国企")),
    ])


def test_split_tags():
    assert split_tags("医疗器械+创口贴+协议转让") == ["医疗器械", "创口贴", "协议转让"]
    assert split_tags("机器人＋减速器+机器人") == ["机器人", "减速器"]  # 全角加号、去重
    assert split_tags(None) == [] and split_tags("-") == [] and split_tags("  ") == [] and split_tags("<NA>") == []


def test_keyword_merge_and_priority():
    tm = _tm()
    for t in ["AI漫剧", "AI短剧", "微短剧", "影视院线"]:
        assert tm.category(t) == ("短剧影视", "题材"), t
    assert tm.category("钙钛矿电池")[0] == "光伏"  # 光伏排在锂电前面，优先匹配
    assert tm.category("固态电池")[0] == "锂电"
    assert tm.category("北京国资") == ("国资国企", "事件")
    assert tm.category("火锅底料") == ("火锅底料", "题材")  # 未归类：原名单独成类


def test_main_category_follows_tag_order():
    tm = _tm()
    # 第一个标签就是主要原因：绿盟科技不会因为当天算力更热就被归到算力
    assert main_category(categories(["人工智能安全", "智算中心", "网络安全"], tm), tm) == "网络安全"
    # 跳过事件类标签（业绩、国资）
    assert main_category(categories(["三季报预增", "AI短剧", "北京国资"], tm), tm) == "短剧影视"
    # 跳过未归类的原名标签，优先归类表里的题材
    assert main_category(categories(["轻资产运营", "数据中心"], tm), tm) == "算力"
    # 只有未归类标签 → 第一个原名；只有事件类 → 第一个事件
    assert main_category(categories(["火锅底料", "业绩增长"], tm), tm) == "火锅底料"
    assert main_category(categories(["业绩增长", "上海国资"], tm), tm) == "业绩"
    assert main_category([], tm) is None


def test_colors_only_for_topic_groups_with_two_or_more():
    tm = _tm()
    counts = Counter({"短剧影视": 5, "算力": 2, "业绩": 4, "光伏": 1})
    colors = assign_colors(counts, tm)
    assert list(colors) == ["短剧影视", "算力"]  # 事件类、只有 1 只的不上色
    assert colors["短剧影视"] == PALETTE[0] and colors["算力"] == PALETTE[1]
    assert category_counts([["短剧影视", "算力"], ["短剧影视"]]) == Counter({"短剧影视": 2, "算力": 1})
    # 没有涨停原因的股票在 pandas 3 里主题材是 NaN（真值）：混进计数时忽略，不能报错
    assert list(assign_colors(Counter({"短剧影视": 2, float("nan"): 3}), tm)) == ["短剧影视"]
    assert list(assign_colors(Counter({"光伏": 1}), tm, min_count=1)) == ["光伏"]


def test_theme_table_ranks_within_theme():
    df = pd.DataFrame({
        "key_t_date": [20261012] * 5 + [20261013],
        "x_t1_rank": [1, 2, 3, 4, 5, 1],     # T-1日序号（首次涨停时间）
        "x_t_rank": [4, 1, 5, 2, 3, 1],      # T日序号（T 日竞价涨幅）
        "x_t1_reason": ["AI短剧+影视院线", "微短剧", "网络安全", "北京国资+AI漫剧", None, "AI短剧"],
    })
    t = theme_table(df, _tm())
    short = [0, 1, 3]  # 同一天的短剧影视：T-1日序号 1、2、4，T日序号 4、1、2
    assert (t.loc[short, "main"] == "短剧影视").all() and t.loc[short, "theme_n"].tolist() == [3, 3, 3]
    assert t.loc[short, "theme_rank_t1"].tolist() == [1, 2, 3]  # 题材内T-1日序号
    assert t.loc[short, "theme_rank_t"].tolist() == [3, 1, 2]   # 题材内T日序号：竞价最强的 1 号排第一
    assert t.loc[2, "main"] == "网络安全" and t.loc[2, "theme_n"] == 1 and t.loc[2, "theme_rank_t"] == 1
    assert pd.isna(t.loc[4, "main"]) and pd.isna(t.loc[4, "theme_rank_t1"]) and pd.isna(t.loc[4, "theme_n"])
    assert t.loc[5, "theme_rank_t1"] == 1 and t.loc[5, "theme_n"] == 1  # 不同 T 日分开排


def test_project_groups_file_loads():
    tm = load_theme_map()
    assert len(tm.groups) >= 30
    assert tm.category("AI漫剧")[0] == tm.category("微短剧")[0] == tm.category("AI短剧")[0] == "短剧影视"
    assert tm.category("BIPV")[0] == "光伏" and tm.category("无锡国资")[0] == "国资国企"
