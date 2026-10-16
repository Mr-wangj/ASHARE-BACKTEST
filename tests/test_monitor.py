"""舆情监控：消息分类、题材映射、涨停原因解析、题材面板、基础模型与催化排序。"""
import numpy as np
import pandas as pd

from ashare_bt.monitor import (Themer, abnormal_news, auc, classify, fit_logit, message_level, message_score,
                               parse_reason, parse_rules, predict, score_messages, theme_panel, theme_points)
from ashare_bt.themes import ThemeMap, parse_groups

RULES = parse_rules({
    "catalyst_beta": 1.0, "important_boost": 2.0,
    "categories": [
        {"name": "政策", "weight": 3, "keywords": ["国务院", "指导意见"]},
        {"name": "海外映射", "weight": 1.5, "keywords": ["Meta", "英伟达"]},
        {"name": "涨价供给", "weight": 2, "keywords": ["涨价", "持续上行"]},
        {"name": "利空", "weight": -2, "keywords": ["减持", "立案"]},
    ],
    "reason_sources": {"研报": ["研报"], "互动易": ["互动易"], "公告": ["公告", "年报"], "媒体": ["报道", "时报"]},
})
TM = ThemeMap(parse_groups({"groups": [
    {"name": "光通信", "keywords": ["CPO", "光模块"]},
    {"name": "有色资源", "keywords": ["铜", "黄金"]},
    {"name": "AI应用", "keywords": ["AI", "大模型"]},
    {"name": "业绩", "kind": "事件", "keywords": ["预增"]},
]}))


def test_classify_and_score():
    assert [c.name for c, _ in classify("Metal 价格涨价", RULES)] == ["涨价供给"]        # Meta 不命中 Metal
    hits = classify("国务院印发指导意见，英伟达涨价", RULES)
    assert [c.name for c, _ in hits] == ["政策", "海外映射", "涨价供给"]
    assert message_score(hits, False, RULES) == ("政策", 3.0)
    assert message_score(classify("股东拟减持", RULES), True, RULES) == ("利空", -4.0)   # 重要快讯加成
    assert message_score([], False, RULES) == (None, 0.0)
    r2 = parse_rules({"categories": [{"name": "会议展会", "weight": 1.5, "keywords": ["大会"], "exclude": ["股东大会"]}]})
    assert classify("世界人工智能大会开幕", r2)[0][0].name == "会议展会"
    assert classify("公司将召开2026年第一次临时股东大会", r2) == []          # 排除词：公司会议不算


def test_themer_rules_for_long_text():
    th = Themer(TM)
    assert th.themes("铜价创新高，CPO 订单饱满") == ["光通信", "有色资源"]
    assert th.themes("铜仁市文旅") == []            # 单字关键词只在“铜价/铜矿/铜业/铜材”里算
    assert th.themes("AI眼镜发布") == []            # 两个字母的英文太宽泛，新闻里不用
    assert th.themes("业绩预增") == []              # 事件类不算题材
    assert th.of_block("共封装光学(CPO)") == "光通信" and th.of_block("冰雪产业") is None
    assert Themer(TM, stop=["铜"]).themes("铜价创新高，黄金走强") == ["有色资源"]   # 停用“铜”后只靠“黄金”命中
    assert Themer(TM, stop=["铜", "黄金"]).themes("铜价创新高，黄金走强") == []


def test_parse_reason_sections_sources_and_freshness():
    text = ("行业原因： 1、据证券时报10月12日报道，铜箔价格持续上行。 2、国金证券9月6日研报指出，需求向好。 "
            "公司原因： 1、据2026年10月12日股票交易异常波动公告，生产经营正常。 2、据2026年8月20日互动易，公司已批量供货。"
            "\n\n（免责声明：本内容由AI生成）")
    items = parse_reason(text, 20261013, RULES)
    got = [(i.section, i.src_kind, i.date, i.fresh) for i in items]
    assert got == [("行业", "媒体", 20261012, True), ("行业", "研报", 20260906, False),
                   ("公司", "公告", 20261012, False),            # 例行异常波动公告：不算催化
                   ("公司", "互动易", 20260820, False)]
    assert items[0].catalysts == ["涨价供给"] and "免责声明" not in items[-1].text
    plain = parse_reason("1、据2026年10月13日公告，公司拟收购某公司。2、公司主营光模块。", 20261013, RULES)
    assert [(i.section, i.src_kind, i.fresh) for i in plain] == [("公司", "公告", True), ("公司", "其他", False)]
    assert parse_reason(None, 20261013, RULES) == []


def test_theme_panel_features_and_label():
    days = [20261008, 20261009, 20261012, 20261013]
    daily = pd.DataFrame({"date": [20261008, 20261009, 20261012, 20261013, 20261012],
                          "theme": ["光通信", "光通信", "光通信", "光通信", "冰雪"],
                          "n_up": [2, 3, 5, 1, 1], "max_h": [1, 2, 3, 4, 1], "n_lb": [0, 1, 2, 1, 0],
                          "early": [1, 1, 3, 0, 0], "n_yizi": [0, 0, 1, 0, 0]})
    mkt = pd.Series(50, index=days)
    p = theme_panel(daily, days, mkt, ["光通信", "有色资源"], TM).set_index(["date", "theme"])
    r = p.loc[(20261012, "光通信")]
    assert (r["n_up"], r["n_prev"], r["hot_days"], r["n_next"], r["y"]) == (5, 3, 3, 1, 0.0)
    assert p.loc[(20261009, "光通信"), "y"] == 1.0                 # 次日 5 家 ≥ 3
    assert p.loc[(20261012, "冰雪"), "n_up"] == 1 and (20261013, "冰雪") not in p.index   # 非归类题材只在有涨停时进候选
    assert p.loc[(20261013, "有色资源"), "n_up"] == 0 and np.isnan(p.loc[(20261013, "有色资源"), "y"])


def test_fit_logit_learns_direction():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 1))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 2 * x[:, 0])))).astype(float)
    w = fit_logit(x, y, l2=0.1)
    assert 1.5 < w[1] < 2.5 and 0.2 < w[0] < 0.8
    assert auc(y, x[:, 0]) > 0.8


def test_message_level_and_points():
    assert message_level("富春染织：预计前三季度净利润增长", "公告,A股", 1) == "个股"
    assert message_level("中国电建：签约贝尔格莱德地铁合同", "A股", 1) == "个股"
    assert message_level("工信部：加快推动AI眼镜产业发展", "A股", 3) == "行业"
    flash = pd.DataFrame({"title": ["工信部印发指导意见", "某某股份：拟减持"], "digest": ["CPO 产业", ""],
                          "important": [False, False], "codes": [[], ["USHA600001"]], "tags": ["A股", "公告,A股"]})
    msgs = score_messages(flash, RULES, Themer(TM), {"USHA600001": "光通信"})
    assert msgs["level"].tolist() == ["行业", "个股"] and msgs["w_theme"].tolist() == [3.0, -2.0 * RULES.company_factor]
    pos, neg = theme_points(msgs)
    assert pos.to_dict() == {"光通信": 3.0} and neg.round(6).to_dict() == {"光通信": round(-2.0 * RULES.company_factor, 6)}
    r2 = parse_rules({"categories": [{"name": "政策", "weight": 3, "keywords": ["指导意见"]}], "ignore_titles": ["收评"]})
    recap = pd.DataFrame({"title": ["A股收评：CPO 涨幅居前，指导意见利好"], "digest": [""], "important": [True],
                          "codes": [[]], "tags": ["A股"]})
    s = score_messages(recap, r2, Themer(TM), {})
    assert s["category"].tolist() == ["行情复盘"] and s["weight"].tolist() == [0.0]   # 行情复盘：只显示、不计分


def test_abnormal_news_removes_the_usual_volume():
    hist = [pd.Series({"医药": 8.0, "光通信": 2.0})] * 3          # 医药平时就占 80%
    abn, base = abnormal_news(pd.Series({"医药": 8.0, "光通信": 8.0}), pd.Series({"医药": -1.0}), hist)
    assert base.round(2).to_dict() == {"医药": 0.8, "光通信": 0.2}
    assert abn["光通信"] > 0 > abn["医药"]                         # 光通信的消息异常增多，医药只是常态（再加利空）
    raw, _ = abnormal_news(pd.Series({"医药": 2.0}), pd.Series(dtype=float), [])
    assert raw.to_dict() == {"医药": 2.0}                           # 没有历史时用原始点数


def test_catalysts_move_ranking():
    model = {"mean": [0] * 7, "std": [1] * 7, "coef": [-1.0, 1.0, 0, 0, 0, 0, 0, 0]}   # 只看今日涨停家数
    panel_day = pd.DataFrame({"date": 20261013, "theme": ["光通信", "有色资源"], "n_up": [3, 2], "n_prev": 0,
                              "max_h": 1, "n_lb": 0, "early": 0, "hot_days": 0, "mkt": 50, "n_next": np.nan, "y": np.nan})
    news = pd.Series({"有色资源": 5.0, "量子": 2.0, "光通信": -2.0})
    pred = predict(20261013, None, panel_day=panel_day, model=model, news=news, events=pd.DataFrame(), hot=None,
                   pop=None, code_theme={}, themer=Themer(TM), rules=RULES).set_index("theme")
    assert pred.loc["有色资源", "news_pts"] == 5.0 and pred.loc["光通信", "news_pts"] == -2.0
    assert pred.index[0] == "有色资源"                       # 催化把今日涨停较少的题材排到前面
    assert pred.loc["量子", "n_up"] == 0                       # 只有消息催化的题材也进表
    assert pred.loc["有色资源", "p_adj"] > pred.loc["有色资源", "p_base"]
