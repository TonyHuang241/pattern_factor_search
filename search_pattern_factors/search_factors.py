import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

# 子进程通过 fork 继承的搜索对象，避免把大块的价格窗口逐个任务序列化传给子进程。
_searcher = None

def _search_pattern(angle):
    return _searcher._search_pattern(angle)

class PatternFactorSearch:
    """搜索三点价格形态，并计算样本内外 IC。"""

    def __init__(self, data, pattern_length, split_date, forward_period, n_jobs):
        self.pattern_length = pattern_length
        self.n_jobs = n_jobs
        self.forward_period = forward_period
        if not isinstance(split_date, (int, np.integer)):
            raise TypeError(f"split_date 必须是日期编号 trade_day（整数），实际传入 {split_date!r}")
        self.split_date = split_date
        self.data = data[["trade_day", "code", "close"]].copy()
        self.data["split"] = np.where(self.data["trade_day"] < self.split_date, "train", "test")
        self.data = self.data.sort_values(["code", "trade_day"])
        self.data = self.data.reset_index(drop=True)
        self.prices = self.data.pivot(index="trade_day", columns="code", values="close")
        self.prices = self.prices.reindex(range(self.prices.index.min(), self.prices.index.max() + 1))
        self.prices = self.prices.where(np.isfinite(self.prices))
        self.split = self.data.drop_duplicates("trade_day").set_index("trade_day")["split"]
        self.split = self.split.reindex(self.prices.index)

    def search(self, num_patterns):
        """均匀枚举形态，按样本内 IC 降序返回结果。"""
        global _searcher
        self.forward_returns = self._forward_returns()
        self.price_windows = self._prepare_windows()
        _searcher = self
        angles = np.linspace(0, 2 * np.pi, num_patterns, endpoint=False)
        with multiprocessing.get_context("fork").Pool(self.n_jobs) as pool:
            records = list(tqdm(pool.imap(_search_pattern, angles), total=num_patterns, desc="search patterns"))

        result = pd.DataFrame(records)
        result = result.sort_values("train_ic", ascending=False, kind="stable", na_position="last")
        result = result.reset_index(drop=True)
        return result

    def _search_pattern(self, angle):
        # 价格窗口和形态都已标准化为零均值、单位长度，二者点积即为 Pearson 相关系数。
        pattern = self._standardize(self._generate_pattern(angle))
        factors = pd.DataFrame(self.price_windows @ pattern, index=self.prices.index, columns=self.prices.columns)
        ic = self._calculate_ic(factors, self.forward_returns)
        return {"angle": angle, "train_ic": ic["train_ic"], "test_ic": ic["test_ic"]}

    def _forward_returns(self):
        forward_returns = self.prices.shift(-self.forward_period) / self.prices - 1
        crosses_split = (self.split == "train") & (self.split.shift(-self.forward_period) == "test")
        forward_returns.loc[crosses_split] = np.nan
        return forward_returns.where(np.isfinite(forward_returns))

    def _calculate_ic(self, factors, forward_returns):
        factors = factors.where(forward_returns.notna())
        forward_returns = forward_returns.where(factors.notna())
        daily_ic = factors.corrwith(forward_returns, axis=1)
        daily_ic = daily_ic.where((factors.max(axis=1) > factors.min(axis=1)) & (forward_returns.max(axis=1) > forward_returns.min(axis=1)))
        return pd.Series({
            "train_ic": daily_ic[self.split == "train"].mean(),
            "test_ic": daily_ic[self.split == "test"].mean(),
        })

    def _generate_pattern(self, angle):
       return np.interp(np.linspace(0, 1, self.pattern_length), [0, 0.5, 1], [np.cos(angle), 0, np.sin(angle)])

    def _standardize(self, values):
        values = values - values.mean(axis=-1, keepdims=True)
        return values / np.sqrt((values ** 2).sum(axis=-1, keepdims=True))

    def _prepare_windows(self):
        prices = np.pad(
            self.prices.to_numpy(dtype=float),
            ((self.pattern_length - 1, 0), (0, 0)),
            constant_values=np.nan,
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            prices, self.pattern_length, axis=0
        )
        windows = np.where(
            windows.max(axis=-1, keepdims=True) > windows.min(axis=-1, keepdims=True),
            windows,
            np.nan,
        )
        return self._standardize(windows)
