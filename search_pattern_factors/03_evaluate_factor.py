import glob
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.offline import get_plotlyjs
from tqdm import tqdm

# 页面样式：内容放在居中的定宽容器里，窄屏时自动缩小
PAGE_STYLE = """
body { margin: 0; background: #f6f6f3; color: #1f1f1d; font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 28px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 8px; }
h2 { font-size: 18px; margin: 32px 0 12px; }
h3 { font-size: 15px; margin: 20px 0 4px; color: #52514e; }
.meta { margin: 0; color: #52514e; line-height: 1.6; }
.card { background: #fff; border: 1px solid #e1e0d9; border-radius: 10px; padding: 16px; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #ecebe6; white-space: nowrap; }
th { color: #52514e; font-weight: 600; }
th[colspan] { text-align: center; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.tabs button { padding: 6px 14px; border: 1px solid #c3c2b7; border-radius: 6px; background: #fff; font: inherit; cursor: pointer; }
.tabs button.active { background: #2a78d6; border-color: #2a78d6; color: #fff; }
"""

# 按钮切换：只显示所选因子的区块，并让其中的图按容器宽度重排；打开页面时默认显示第一个因子
PAGE_SCRIPT = """
function showFactor(id) {
  document.querySelectorAll(".factor").forEach(function (section) { section.hidden = section.id !== "factor-" + id; });
  document.querySelectorAll(".tabs button").forEach(function (button) { button.classList.toggle("active", button.dataset.id === id); });
  document.querySelectorAll("#factor-" + id + " .plotly-graph-div").forEach(function (plot) { Plotly.Plots.resize(plot); });
}
showFactor(document.querySelector(".tabs button").dataset.id);
"""

if __name__ == "__main__":
    config = json.load(open("config/config_pattern_search.json", encoding="utf-8"))
    config_local = json.load(open("config_local.json", encoding="utf-8"))
    config = {key: os.path.join(config_local["root_dir"], value) if key.endswith(("_dir", "_file")) else value for key, value in config.items()}
    os.makedirs(config["output_dir"], exist_ok=True)
    # 清空上次输出的评估报告
    for file in glob.glob(os.path.join(config["output_dir"], "factor_evaluation_*.html")):
        os.remove(file)

    # 10 点开盘价乘复权因子得到后复权价，停牌日置为 NaN，透视成交易日 × 股票的宽表
    daily = pd.read_parquet(config["stock_daily_file"], columns=["date", "code", "adj_factor", "is_suspended"])
    price_10am = pd.read_parquet(config["stock_10am_file"], columns=["code", "trade_time", "open"])
    price_10am["date"] = price_10am["trade_time"].dt.strftime("%Y%m%d")
    daily = daily.merge(price_10am[["code", "date", "open"]], on=["code", "date"], how="left")
    daily["open_10am"] = (daily["open"] * daily["adj_factor"]).where(~daily["is_suspended"])
    open_10am = daily.pivot(index="date", columns="code", values="open_10am")
    # 因子在当日收盘后得到，收益从下一日 10 点算起：forward_return 持有 forward_period 日用于算 IC，next_return 持有 1 日用于分层
    start, split = open_10am.index.searchsorted([config["start_date"], config["split_date"]])
    forward_return = (open_10am.shift(-config["forward_period"] - 1) / open_10am.shift(-1) - 1).iloc[start:]
    next_return = (open_10am.shift(-2) / open_10am.shift(-1) - 1).iloc[start:]
    stock_return = pd.DataFrame({"forward_return": forward_return.stack(), "next_return": next_return.stack()}).reset_index()
    # 两个区间的起止日期，样本内去掉未来收益跨到样本外的最后 forward_period + 1 天，与 02_calculate_factor_value.py 一致
    periods = {"样本内": (open_10am.index[start], open_10am.index[split - config["forward_period"] - 2]), "样本外": (open_10am.index[split], open_10am.index[-1])}

    # 分层曲线颜色：第 1 组深蓝渐浅、最后一组深红，相邻组易区分且能看出顺序
    colors = sample_colorscale(["#00326c", "#03528e", "#2f72b1", "#5194d5", "#73b6fa", "#fc8f6e", "#d66d4c", "#b04b2b", "#8c2901", "#680000"], config["num_groups"])

    for factor_file in sorted(glob.glob(os.path.join(config["factor_value_dir"], "factor_value_*.parquet"))):
        tag = os.path.basename(factor_file).removeprefix("factor_value_").removesuffix(".parquet")
        n, window_length = tag.removeprefix("n").split("_len")
        factor_ic = pd.read_parquet(os.path.join(config["factor_value_dir"], f"factor_ic_{tag}.parquet"))
        weight = pd.read_parquet(os.path.join(config["random_weight_dir"], f"random_weight_{tag}.parquet")).set_index("weight_id")
        weight.columns = range(weight.shape[1] - 1, -1, -1)  # 列名改为距今交易日数，0 为最近一天
        # 因子值文件只含当日在股票池内的股票，以它为底表左连接收益，后面的 IC 和分层都只在股票池内计算
        factor = pd.read_parquet(factor_file)
        factor = factor.merge(stock_return, on=["code", "date"], how="left")
        factor_columns = [column for column in factor.columns if column.startswith("factor_")]
        weight_ids = [int(column.removeprefix("factor_")) for column in factor_columns]

        # 每天的截面 IC 和 RankIC，口径与 02_calculate_factor_value.py 相同：只用未来收益有效的股票
        ic, rank_ic = [], []
        for date, group in tqdm(factor.dropna(subset=["forward_return"]).groupby("date"), desc=f"{tag} IC 和 RankIC", unit="天"):
            ic.append(group[factor_columns].corrwith(group["forward_return"]).rename(date))
            rank_ic.append(group[factor_columns].rank().corrwith(group["forward_return"].rank()).rename(date))
        ic = pd.DataFrame(ic)
        rank_ic = pd.DataFrame(rank_ic)

        # 每个入选因子一个区块：权重形状、样本内外分层累计收益
        factor = factor.dropna(subset=["next_return"])
        factor["count"] = factor.groupby("date")["next_return"].transform("count")
        monotonicity = {"样本内": [], "样本外": []}
        sections = []
        for i, column in enumerate(tqdm(factor_columns, desc=f"{tag} 分层收益", unit="个因子")):
            # 权重形状：窗口内每天一根柱子，正值红色、负值蓝色
            weight_figure = go.Figure()
            weight_figure.add_trace(go.Bar(x=weight.columns, y=weight.loc[weight_ids[i]], marker=dict(color=np.where(weight.loc[weight_ids[i]] > 0, "#d03b3b", "#2a78d6"), line=dict(width=0)), showlegend=False, hovertemplate="距今 %{x} 个交易日<br>权重 %{y:.4f}<extra></extra>"))
            weight_figure.update_layout(template="plotly_white", height=320, margin=dict(l=60, r=20, t=10, b=50), bargap=0.15, xaxis=dict(title="距今交易日数（0 为最近一天）", autorange="reversed"), yaxis=dict(title="权重", zerolinecolor="#898781"))

            # 每天按因子值从小到大等分 num_groups 组（第 1 组最小），组收益为组内股票 next_return 的等权平均
            factor["group"] = ((factor.groupby("date")[column].rank(method="first") - 1) * config["num_groups"] // factor["count"] + 1).astype(int)
            group_return = factor.groupby(["date", "group"])["next_return"].mean().unstack()
            # 单调性：各组在区间内的累计收益与组号的 Pearson 相关系数
            for period, (first_date, last_date) in periods.items():
                monotonicity[period].append(np.corrcoef(group_return.columns, (1 + group_return.loc[first_date:last_date]).prod() - 1)[0, 1])
            # 累计收益曲线样本内外连着画，红色虚线为样本外第一天
            cumulative_return = (1 + group_return).cumprod() - 1
            return_figure = go.Figure()
            for group in cumulative_return.columns:
                return_figure.add_trace(go.Scatter(x=pd.to_datetime(cumulative_return.index, format="%Y%m%d"), y=cumulative_return[group], mode="lines", name=f"第 {group} 组", line=dict(width=1.8, color=colors[group - 1])))
            return_figure.add_vline(x=pd.to_datetime(periods["样本外"][0], format="%Y%m%d"), line=dict(color="red", width=1.5, dash="dash"))
            return_figure.update_layout(template="plotly_white", height=460, margin=dict(l=60, r=20, t=10, b=50), hovermode="x unified", legend=dict(title=f"第 1 组因子值最小<br>第 {config['num_groups']} 组因子值最大"), xaxis=dict(title="因子日期", hoverformat="%Y-%m-%d"), yaxis=dict(title="累计收益", tickformat=".0%", hoverformat=".2%"))
            sections.append(f'<section class="factor" id="factor-{weight_ids[i]}"><h3>权重形状（窗口内每天一根柱子，红色为正、蓝色为负）</h3>{weight_figure.to_html(full_html=False, include_plotlyjs=False)}<h3>分层累计收益（每天按因子值等分 {config["num_groups"]} 组，下一日 10 点等权买入、再下一日 10 点卖出；红色虚线左侧为样本内，右侧为样本外）</h3>{return_figure.to_html(full_html=False, include_plotlyjs=False)}</section>')

        # 入选因子按 02 的挑选顺序排列，样本内外各算一套指标，t 值为日度序列的均值 / 标准差 × √天数
        summary = pd.DataFrame({("", "因子"): weight_ids}, index=factor_columns)
        for period, (first_date, last_date) in periods.items():
            period_ic = ic.loc[first_date:last_date]
            period_rank_ic = rank_ic.loc[first_date:last_date]
            summary[(period, "IC")] = period_ic.mean().map("{:.4f}".format)
            summary[(period, "RankIC")] = period_rank_ic.mean().map("{:.4f}".format)
            summary[(period, "IC t 值")] = (period_ic.mean() / period_ic.std() * np.sqrt(period_ic.count())).map("{:.2f}".format)
            summary[(period, "RankIC t 值")] = (period_rank_ic.mean() / period_rank_ic.std() * np.sqrt(period_rank_ic.count())).map("{:.2f}".format)
            summary[(period, "分组单调性")] = pd.Series(monotonicity[period], index=factor_columns).map("{:.3f}".format)
        print(f"{tag}，IC 天数 {len(ic)}，最后一个因子日期 {ic.index[-1]}\n{summary.to_string(index=False)}")

        # 页面：标题、说明、指标表，以及按钮切换的单因子区块
        buttons = "".join([f'<button data-id="{weight_id}" onclick="showFactor(this.dataset.id)">因子 {weight_id}</button>' for weight_id in weight_ids])
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>随机权重因子评估 {tag}</title><style>{PAGE_STYLE}</style><script>{get_plotlyjs()}</script></head>
<body><main>
<h1>随机权重因子评估</h1>
<p class="meta">n = {n}，窗口 {window_length} 天；股票池为指数 {config["stock_pool_index_code"]} 成分股；样本内 {periods["样本内"][0]} 至 {periods["样本内"][1]}，样本外 {periods["样本外"][0]} 至 {ic.index[-1]}（均为因子日期）；共 {len(factor_ic)} 个因子，入选样本内 |RankIC| 最大的 {len(weight_ids)} 个</p>
<h2>入选因子指标（按样本内 |RankIC| 从大到小）</h2>
<p class="meta">因子值在当日收盘后得到；IC 和 RankIC 用下一日 10 点开盘价到其后第 {config["forward_period"]} 个交易日 10 点开盘价的后复权收益计算，取区间内日度均值；t 值 = 日度均值 / 日度标准差 × √天数；分组单调性为每天按因子值等分 {config["num_groups"]} 组后，各组区间累计收益与组号 1 至 {config["num_groups"]} 的 Pearson 相关系数</p>
<div class="card" style="margin-top: 12px;">{summary.to_html(index=False, border=0)}</div>
<h2>单因子详情</h2>
<div class="tabs">{buttons}</div>
<div class="card">{''.join(sections)}</div>
</main><script>{PAGE_SCRIPT}</script></body>
</html>"""
        with open(os.path.join(config["output_dir"], f"factor_evaluation_{tag}.html"), "w", encoding="utf-8") as file:
            file.write(html)
