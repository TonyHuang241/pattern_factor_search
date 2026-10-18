import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

class PatternFactorSearch:
    def __init__(self, data, pattern_length, split_date, forward_period):
        self.pattern_length = pattern_length
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

    def search(self, num_patterns, desc="search patterns"):
        """均匀枚举形态，按样本内 IC 降序返回结果。"""
        print(f"device: {"cuda" if torch.cuda.is_available() else "cpu"}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prices = torch.tensor(self.prices.to_numpy(dtype=np.float32), device=device)
        forward_returns = self._forward_returns(prices)
        price_windows = self._prepare_windows(prices)
        records = []
        for angle in tqdm(np.linspace(0, 2 * np.pi, num_patterns, endpoint=False), desc=desc):
            # 价格窗口和形态都已标准化为零均值、单位长度，二者点积即为 Pearson 相关系数。
            pattern = self._standardize(self._generate_pattern(angle, device))
            ic = self._calculate_ic(price_windows @ pattern, forward_returns)
            records.append({"angle": angle, "train_ic": ic["train_ic"], "test_ic": ic["test_ic"]})

        result = pd.DataFrame(records)
        result = result.sort_values("train_ic", ascending=False, kind="stable", na_position="last")
        result = result.reset_index(drop=True)
        return result

    def _forward_returns(self, prices):
        forward_returns = F.pad(
            prices[self.forward_period:], 
            (0, 0, 0, self.forward_period), 
            value=torch.nan
        ) / prices - 1
        crosses_split = (self.split == "train") & (self.split.shift(-self.forward_period) == "test")
        forward_returns[torch.tensor(crosses_split.to_numpy(), device=prices.device)] = torch.nan
        forward_returns = torch.where(forward_returns.isfinite(), forward_returns, torch.nan)
        return forward_returns

    def _calculate_ic(self, factors, forward_returns):
        factors = torch.where(forward_returns.isnan(), torch.nan, factors)
        forward_returns = torch.where(factors.isnan(), torch.nan, forward_returns)
        
        valid = (torch.where(factors.isnan(), -torch.inf, factors).amax(dim=1) > torch.where(factors.isnan(), torch.inf, factors).amin(dim=1)) & (torch.where(forward_returns.isnan(), -torch.inf, forward_returns).amax(dim=1) > torch.where(forward_returns.isnan(), torch.inf, forward_returns).amin(dim=1))
        factors = factors - factors.nanmean(dim=1, keepdim=True)
        forward_returns = forward_returns - forward_returns.nanmean(dim=1, keepdim=True)
        
        daily_ic = (factors * forward_returns).nansum(dim=1) / torch.sqrt((factors ** 2).nansum(dim=1) * (forward_returns ** 2).nansum(dim=1))
        daily_ic = torch.where(valid, daily_ic, torch.nan)

        return pd.Series({
            "train_ic": daily_ic[torch.tensor((self.split == "train").to_numpy(), device=daily_ic.device)].nanmean().item(),
            "test_ic": daily_ic[torch.tensor((self.split == "test").to_numpy(), device=daily_ic.device)].nanmean().item(),
        })

    def _generate_pattern(self, angle, device):
        x = torch.linspace(0, 1, self.pattern_length, dtype=torch.float32, device=device)
        return torch.where(x <= 0.5, np.cos(angle) * (1 - 2 * x), np.sin(angle) * (2 * x - 1))

    def _standardize(self, values):
        values = values - values.mean(dim=-1, keepdim=True)
        return values / torch.sqrt((values ** 2).sum(dim=-1, keepdim=True))

    def _prepare_windows(self, prices):
        prices = torch.nn.functional.pad(prices, (0, 0, self.pattern_length - 1, 0), value=torch.nan)
        # unfold 得到 (日期, 股票, 窗口) 的滑动窗口视图，第 t 行是截至第 t 天的 pattern_length 个价格。
        windows = prices.unfold(0, self.pattern_length, 1)
        windows = torch.where(windows.amax(dim=-1, keepdim=True) > windows.amin(dim=-1, keepdim=True), windows, torch.nan)
        return self._standardize(windows)
