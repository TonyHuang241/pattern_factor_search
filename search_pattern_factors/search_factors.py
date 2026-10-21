import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

if __package__:
    from .price_compression import AutoEncoder
else:
    from price_compression import AutoEncoder

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_LOCAL_FILE = PROJECT_DIR / "config_local.json"
CONFIG_FILE = PROJECT_DIR / "config/config_pattern_search.json"
SEARCH_CLASS = "AutoencoderPatternFactorSearch"
PATTERN_LENGTH = 60
LATENT_DIM = 8
NUM_PATTERNS = 500
RANDOM_SEED = 42
MODEL_DIR = PROJECT_DIR / "model"

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
        return self._search(np.linspace(0, 2 * np.pi, num_patterns, endpoint=False), "angle", desc)

    def _search(self, points, point_name, desc):
        print(f"device: {"cuda" if torch.cuda.is_available() else "cpu"}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prices = torch.tensor(self.prices.to_numpy(dtype=np.float32), device=device)
        forward_returns = self._forward_returns(prices)
        price_windows = self._prepare_windows(prices)
        records = []
        for point in tqdm(points, desc=desc):
            # 价格窗口和形态都已标准化为零均值、单位长度，二者点积即为 Pearson 相关系数。
            pattern = self._standardize(self._generate_pattern(point, device))
            ic = self._calculate_ic(price_windows @ pattern, forward_returns)
            records.append({point_name: point.tolist(), "train_ic": ic["train_ic"], "test_ic": ic["test_ic"]})

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

class AutoencoderPatternFactorSearch(PatternFactorSearch):
    def __init__(self, data, pattern_length, split_date, forward_period, latent_dim, model_dir=MODEL_DIR, seed=RANDOM_SEED):
        super().__init__(data, pattern_length, split_date, forward_period)
        self.latent_dim = latent_dim
        self.model_dir = Path(model_dir)
        self.seed = seed
        self.model = None

    def search(self, num_patterns, desc="search patterns"):
        """在潜在空间单位球面上均匀随机采样，按样本内 IC 降序返回结果。"""
        points = np.random.default_rng(self.seed).standard_normal((num_patterns, self.latent_dim))
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        return self._search(points, "latent_point", desc)

    def _generate_pattern(self, point, device):
        if self.model is None:
            self.model = AutoEncoder(self.pattern_length, self.latent_dim).to(device)
            self.model.load_state_dict(torch.load(self.model_dir / f"autoencoder_len{self.pattern_length}_dim{self.latent_dim}.pt", map_location=device, weights_only=True))
            self.model.eval()
        with torch.no_grad():
            return self.model.decoder(torch.as_tensor(point, dtype=torch.float32, device=device))


def plot_result(result, output_file, title):
    """按切分日绘制样本内外 IC 散点图，所有子图的两轴范围统一且关于零对称。"""
    groups = result.groupby("split_date")
    axis_limit = max(0.01, result[["train_ic", "test_ic"]].abs().max().max() * 1.05)
    ncols = min(3, len(groups))
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True, squeeze=False, layout="constrained")
    for ax, (split_date, group) in zip(axes.flat, groups):
        ax.scatter(group["train_ic"], group["test_ic"], s=8, color="#2a78d6", alpha=0.7, linewidths=0)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.axvline(0, color="#52514e", linewidth=0.8)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MultipleLocator(0.01))
            axis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(color="#e5e4e0", linewidth=0.5)
        ax.set(xlabel="train_ic", ylabel="test_ic", title=str(split_date), xlim=(-axis_limit, axis_limit), ylim=(-axis_limit, axis_limit))
    for ax in axes.flat[len(groups):]:
        ax.remove()
    fig.suptitle(title)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_search(config):
    """读取日线数据，滚动搜索形态因子并保存 IC 散点图。"""
    search_class = {"PatternFactorSearch": PatternFactorSearch, "AutoencoderPatternFactorSearch": AutoencoderPatternFactorSearch}[SEARCH_CLASS]
    search_kwargs = {}
    run_name = f"len{PATTERN_LENGTH}_dim{LATENT_DIM}"
    if search_class is AutoencoderPatternFactorSearch:
        search_kwargs = {"latent_dim": LATENT_DIM, "model_dir": MODEL_DIR, "seed": RANDOM_SEED}
    output_dir = Path(config["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_daily = pd.read_parquet(config["stock_daily_file"])
    trade_dates = stock_daily.drop_duplicates("trade_day").set_index("trade_day")["date"]

    first_split_date = stock_daily.loc[stock_daily["date"] >= config["start_date"], "trade_day"].min()
    results = []
    for split_date in range(first_split_date, stock_daily["trade_day"].max() + 1, config["roll_step_length"]):
        pattern_searcher = search_class(
            data=stock_daily[(stock_daily["trade_day"] >= split_date - config["train_length"]) & (stock_daily["trade_day"] < split_date + config["test_length"])],
            pattern_length=PATTERN_LENGTH,
            split_date=split_date,
            forward_period=5,
            **search_kwargs,
        )
        result = pattern_searcher.search(num_patterns=NUM_PATTERNS, desc=f"split date {trade_dates[split_date]}")
        if torch.cuda.is_available():
            # search 返回后局部 GPU 张量已释放，清理未使用的显存缓存。
            torch.cuda.empty_cache()
        result["split_date"] = trade_dates[split_date]
        results.append(result)
    results = pd.concat(results, ignore_index=True)
    plot_result(results, output_dir / "pattern_ic.png", f"{run_name}: split date {config['start_date']} onwards")
    results.to_parquet(output_dir / "results.parquet", index=False)


if __name__ == "__main__":
    config = json.load(open(CONFIG_FILE, "r"))
    config_local = json.load(open(CONFIG_LOCAL_FILE, "r"))
    config = {k: config_local["root_dir"] + v if k.split('_')[-1] in ["file", "dir"] else v for k, v in config.items()}

    run_search(config)
