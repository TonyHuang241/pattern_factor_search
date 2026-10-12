"""三点形态的枚举及滚动相关性因子计算。"""

import numpy as np
import pandas as pd


class PatternFactorSearch:
    """搜索三点价格形态，并计算样本内外 IC。"""

    def __init__(self, data, pattern_length, split_date, forward_period):
        self.pattern_length = pattern_length
        self.forward_period = forward_period
        self.split_date = pd.Timestamp(split_date)
        self.data = data[["date", "trade_day", "code", "close"]].copy()
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data["split"] = np.where(self.data["date"] < self.split_date, "train", "test")
        self.data = self.data.sort_values(["code", "trade_day"])
        self.data = self.data.reset_index(drop=True)
        self.prices = self.data.pivot(index="trade_day", columns="code", values="close")
        self.prices = self.prices.reindex(range(self.prices.index.min(), self.prices.index.max() + 1))
        self.prices = self.prices.where(np.isfinite(self.prices))
        self.split = self.data.drop_duplicates("trade_day").set_index("trade_day")["split"]
        self.split = self.split.reindex(self.prices.index)

    def search(self, num_patterns = 360):
        """均匀枚举形态，按样本内 IC 降序返回结果。"""
        forward_returns = self._forward_returns()
        price_centered, price_norm = self._prepare_windows()
        records = []
        for angle in np.linspace(0, 2 * np.pi, num_patterns, endpoint=False):
            pattern = np.interp(
                np.linspace(0, 1, self.pattern_length),
                [0, 0.5, 1],
                [np.cos(angle), 0, np.sin(angle)],
            )
            pattern_centered = pattern - pattern.mean()
            # Pearson r = Σ[(价格 - 均价)(形态 - 均值)] / √[Σ(价格 - 均价)² Σ(形态 - 均值)²]
            numerator = price_centered @ pattern_centered  # 沿窗口求乘积和。
            denominator = price_norm * np.sqrt((pattern_centered ** 2).sum())
            factors = pd.DataFrame(numerator / denominator, index=self.prices.index, columns=self.prices.columns)
            ic = self._calculate_ic(factors, forward_returns)
            records.append({"angle": angle, "train_ic": ic["train_ic"], "test_ic": ic["test_ic"]})
        result = pd.DataFrame(records)
        result = result.sort_values("train_ic", ascending=False, kind="stable", na_position="last")
        return result.reset_index(drop=True)

    def _forward_returns(self):
        forward_returns = self.prices.shift(-self.forward_period) / self.prices - 1
        # 训练收益标签不能跨入测试期。
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

    def _prepare_windows(self):
        prices = np.pad(
            self.prices.to_numpy(dtype=float), 
            ((self.pattern_length - 1, 0), (0, 0)), 
            constant_values=np.nan
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            prices, self.pattern_length, axis=0
        )
        price_centered = windows - windows.mean(axis=-1, keepdims=True)
        price_norm = np.sqrt((price_centered ** 2).sum(axis=-1))
        price_norm[windows.max(axis=-1) == windows.min(axis=-1)] = np.nan
        return price_centered, price_norm
