import os, json, shutil

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from search_pattern_factors.search_factors import PatternFactorSearch

CONFIG_LOCAL_FILE = "config_local.json"
CONFIG_FILE = "config.json"

def plot_result(result, output_file, title):
    """绘制所有切分日每个形态的样本内 IC（横轴）与样本外 IC（纵轴）散点图，每个切分日一种颜色和点形。"""
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#0d366b", "#8c5a2b", "#52514e", "#9a9a1a"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (split_date, group) in enumerate(result.groupby("split_date")):
        ax.scatter(group["train_ic"], group["test_ic"], s=8, color=colors[i % len(colors)], marker=markers[i % len(markers)], alpha=0.7, linewidths=0, label=split_date)
    legend = ax.legend(title="split date", loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=2.5, frameon=False)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.grid(color="#e5e4e0", linewidth=0.5)
    ax.set(xlabel="train_ic", ylabel="test_ic", title=title)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    config = json.load(open(CONFIG_FILE, "r"))
    config_local = json.load(open(CONFIG_LOCAL_FILE, "r"))

    shutil.rmtree(config["output_dir"], ignore_errors=True)
    os.makedirs(config["output_dir"])

    config = {k: config_local["root_dir"] + v if k.split('_')[-1] in ["file", "dir"] else v for k, v in config.items()}

    stock_daily = pd.read_parquet(config["stock_daily_file"])
    trade_dates = stock_daily.drop_duplicates("trade_day").set_index("trade_day")["date"]

    first_split_date = stock_daily.loc[stock_daily["date"] >= config["start_date"], "trade_day"].min()
    results = []
    for split_date in range(first_split_date, stock_daily["trade_day"].max() + 1, config["roll_step_length"]):
        pattern_searcher = PatternFactorSearch(
            data=stock_daily[(stock_daily["trade_day"] >= split_date - config["train_length"]) & (stock_daily["trade_day"] < split_date + config["test_length"])],
            pattern_length=config["pattern_length"],
            split_date=split_date,
            forward_period=5,
        )
        result = pattern_searcher.search(num_patterns=360, desc=f"split date {trade_dates[split_date]}")
        result["split_date"] = trade_dates[split_date]
        results.append(result)
    plot_result(pd.concat(results), os.path.join(config["output_dir"], "pattern_ic.png"), f"split date {config['start_date']} onwards")


    