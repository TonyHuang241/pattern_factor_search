import glob
import json
import os

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_rank_ic(returns, rolling_mean, forward_return, weight):
    """逐日计算全部因子与未来收益的截面 RankIC，returns 和 rolling_mean 比 forward_return 多出前 window_length - 1 天历史，返回 (天数, 因子数)。"""
    # 每天的回报升维成 6 维：回报、绝对值、符号、平方、三次方、滚动平均
    embedding = np.stack([returns, np.abs(returns), np.sign(returns), returns ** 2, returns ** 3, rolling_mean])
    rank_ic = np.zeros((len(forward_return), len(weight)))
    for day in range(len(forward_return)):
        # 只用窗口内 6 个维度都没有 NaN 且未来收益有效的股票
        window = embedding[:, day : day + weight.shape[1]]
        valid = ~np.isnan(window).any(axis=(0, 1)) & ~np.isnan(forward_return[day])
        window = window[:, :, valid]
        # 各维度在窗口内沿时间做 zscore，6 个维度平均后与权重点积，等价于 6 个点积取平均
        mean = window.mean(axis=1, keepdims=True)
        std = window.std(axis=1, keepdims=True)
        # 常数序列因浮点误差标准差可能略大于 0，按相对阈值视为常数，标准化后置 0
        std[std <= 1e-10 * np.abs(mean)] = np.inf
        factor = ((window - mean) / std).mean(axis=0).T @ weight.T
        # 因子值和未来收益都换成截面排名（并列取平均），排名的 Pearson 相关即 RankIC
        factor = pd.DataFrame(factor).rank().to_numpy()
        factor = factor - factor.mean(axis=0)
        target = pd.Series(forward_return[day, valid]).rank().to_numpy()
        target = target - target.mean()
        rank_ic[day] = target @ factor / np.sqrt((factor ** 2).sum(axis=0) * (target @ target))
    return rank_ic


if __name__ == "__main__":
    config = json.load(open("config/config_pattern_search.json", encoding="utf-8"))
    config_local = json.load(open("config_local.json", encoding="utf-8"))
    config = {key: os.path.join(config_local["root_dir"], value) if key.endswith(("_dir", "_file")) else value for key, value in config.items()}
    os.makedirs(config["factor_value_dir"], exist_ok=True)
    # 清空上次输出的 RankIC 和因子值文件
    for file in glob.glob(os.path.join(config["factor_value_dir"], "factor_*.parquet")):
        os.remove(file)

    # 10 点开盘价乘复权因子得到后复权价，停牌日的收盘价和 10 点价都置为 NaN
    daily = pd.read_parquet(config["stock_daily_file"], columns=["date", "code", "close", "adj_factor", "is_suspended"])
    price_10am = pd.read_parquet(config["stock_10am_file"], columns=["code", "trade_time", "open"])
    price_10am["date"] = price_10am["trade_time"].dt.strftime("%Y%m%d")
    daily = daily.merge(price_10am[["code", "date", "open"]], on=["code", "date"], how="left")
    daily["close"] = daily["close"].where(~daily["is_suspended"])
    daily["open_10am"] = (daily["open"] * daily["adj_factor"]).where(~daily["is_suspended"])
    # 透视成交易日 × 股票的宽表；因子在当日收盘后得到，未来收益从下一日 10 点算到再过 forward_period 日的 10 点
    close = daily.pivot(index="date", columns="code", values="close")
    open_10am = daily.pivot(index="date", columns="code", values="open_10am")
    returns = close / close.shift(1) - 1
    # 含当日在内过去 rolling_mean_window 日的回报均值，只向过去看，窗口内有 NaN 则为 NaN
    rolling_mean = returns.rolling(config["rolling_mean_window"]).mean()
    forward_return = open_10am.shift(-config["forward_period"] - 1) / open_10am.shift(-1) - 1

    # 股票池：每个交易日取不晚于当日的最近一期指数成分股，池外股票的未来收益置为 NaN，RankIC 只在池内计算
    index_stock = pd.read_parquet(config["index_stock_file"], filters=[("index_code", "==", config["stock_pool_index_code"])])
    stock_pool = index_stock.pivot(index="date", columns="code", values="index_code")
    stock_pool = stock_pool.reindex(close.index, method="ffill").reindex(columns=close.columns)
    stock_pool = stock_pool.notna()
    forward_return = forward_return.where(stock_pool)

    # 样本内按连续交易日分块，去掉未来收益跨到样本外的最后 forward_period + 1 天；块数多于进程数，进度条才会平滑推进
    start, split = close.index.searchsorted([config["start_date"], config["split_date"]])
    chunks = np.array_split(np.arange(start, split - config["forward_period"] - 1), config["num_processes"] * 4)

    with Pool(config["num_processes"]) as pool:
        for weight_file in sorted(glob.glob(os.path.join(config["random_weight_dir"], "random_weight_*.parquet"))):
            tag = os.path.basename(weight_file).removeprefix("random_weight_").removesuffix(".parquet")
            weight = pd.read_parquet(weight_file).set_index("weight_id")
            window_length = weight.shape[1]  # 窗口长度以权重文件的列数为准

            # 每块带上前 window_length - 1 天收益和滚动平均作为窗口历史（滚动平均已在全序列上算好），worker 只返回每天每个因子的 RankIC
            tasks = []
            for chunk in chunks:
                tasks.append(pool.apply_async(calculate_rank_ic, (returns.iloc[chunk[0] - window_length + 1 : chunk[-1] + 1].to_numpy(), rolling_mean.iloc[chunk[0] - window_length + 1 : chunk[-1] + 1].to_numpy(), forward_return.iloc[chunk[0] : chunk[-1] + 1].to_numpy(), weight.to_numpy())))
            rank_ic = []
            for task in tqdm(tasks, desc=f"{tag} 样本内 RankIC", unit="块"):
                rank_ic.append(task.get())
            rank_ic = pd.DataFrame(np.vstack(rank_ic), columns=weight.index)
            factor_ic = pd.DataFrame({"rank_ic": rank_ic.mean(), "rank_icir": rank_ic.mean() / rank_ic.std()}).reset_index()
            factor_ic.to_parquet(os.path.join(config["factor_value_dir"], f"factor_ic_{tag}.parquet"), index=False)

            # 挑选样本内 |RankIC| 最大的 num_top_factors 个因子，从大到小排列
            top = factor_ic.loc[factor_ic["rank_ic"].abs().nlargest(config["num_top_factors"]).index, "weight_id"]

            # 重算入选因子在 start_date 之后全部交易日的因子值，只输出当日在股票池内且窗口内 6 个维度都没有 NaN 的股票
            history = returns.iloc[start - window_length + 1 :].to_numpy()
            embedding = np.stack([history, np.abs(history), np.sign(history), history ** 2, history ** 3, rolling_mean.iloc[start - window_length + 1 :].to_numpy()])
            day_index, stock_index = np.nonzero(~np.lib.stride_tricks.sliding_window_view(np.isnan(embedding).any(axis=0), window_length, axis=0).any(axis=2) & stock_pool.iloc[start:].to_numpy())
            # 逐个维度摊开有效窗口做 zscore 并累加，控制内存，口径与 calculate_rank_ic 一致
            factor_value = 0
            for dimension in embedding:
                windows = np.lib.stride_tricks.sliding_window_view(dimension, window_length, axis=0)[day_index, stock_index]
                mean = windows.mean(axis=1, keepdims=True)
                std = windows.std(axis=1, keepdims=True)
                std[std <= 1e-10 * np.abs(mean)] = np.inf
                factor_value = factor_value + (windows - mean) / std
            factor_value = pd.DataFrame(((factor_value / len(embedding)) @ weight.loc[top].to_numpy().T).astype(np.float32), columns=[f"factor_{weight_id}" for weight_id in top])
            factor_value.insert(0, "code", close.columns[stock_index])
            factor_value.insert(1, "date", close.index[start + day_index])
            factor_value.to_parquet(os.path.join(config["factor_value_dir"], f"factor_value_{tag}.parquet"), index=False)
            print(f"{tag}：{len(weight)} 个因子样本内 RankIC 计算完成（{len(rank_ic)} 天），入选 {len(top)} 个因子 {top.tolist()}")
