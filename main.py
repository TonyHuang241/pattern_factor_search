import os, json

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from search_pattern_factors.search_factors import PatternFactorSearch

CONFIG_LOCAL_FILE = "config_local.json"
CONFIG_FILE = "config.json"

def plot_result(result, output_file, title):
    """绘制每个形态的样本内 IC（横轴）与样本外 IC（纵轴）散点图。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(result["train_ic"], result["test_ic"], s=16, color="#2a78d6", edgecolors="#fcfcfb", linewidths=0.5)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.grid(color="#e5e4e0", linewidth=0.5)
    ax.set(xlabel="train_ic", ylabel="test_ic", title=title)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    config = json.load(open(CONFIG_FILE, "r"))
    config_local = json.load(open(CONFIG_LOCAL_FILE, "r"))

    config = {k: config_local["root_dir"] + v if k.split('_')[-1] in ["file", "dir"] else v for k, v in config.items()}

    stock_daily = pd.read_parquet(config["stock_daily_file"])
    stock_daily["year"] = stock_daily["date"].str[:4].astype(int)
    year_start = stock_daily.groupby("year")["trade_day"].min().to_dict()
    for year in range(2015, 2027):
        pattern_searcher = PatternFactorSearch(
            data=stock_daily[stock_daily["year"].isin([year - 1, year])],
            pattern_length=20,
            split_date=year_start[year],
            forward_period=1,
            n_jobs=config["n_jobs"],
        )
        result = pattern_searcher.search(num_patterns=360)
        plot_result(result, os.path.join(config["output_dir"], f"pattern_ic_{year}.png"), f"split year {year}")


    