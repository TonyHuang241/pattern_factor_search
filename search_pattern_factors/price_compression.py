import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, in_dim),
        )

    def encode(self, x):
        """将已完成 Min-Max 标准化的窗口压缩为单位长度向量。"""
        return F.normalize(self.encoder(x), p=2, dim=-1)

    def forward(self, x):
        return self.decoder(self.encode(x))


class PriceWindowDataset(Dataset):
    """仅保存 CPU 价格矩阵和窗口索引，取样时生成标准化窗口。"""

    def __init__(self, data, pattern_length):
        if pattern_length < 2:
            raise ValueError("pattern_length 必须至少为 2")
        self.pattern_length = pattern_length
        self.first_trade_day = int(data["trade_day"].min())
        data = data[["code", "trade_day", "close", "is_suspended"]].copy()
        data["close"] = data["close"].astype(np.float32)
        data.loc[data["is_suspended"], "close"] = np.nan
        data = data.pivot(index="code", columns="trade_day", values="close")
        data = data.reindex(columns=range(self.first_trade_day, int(data.columns.max()) + 1))
        self.prices = np.ascontiguousarray(data.to_numpy(dtype=np.float32))

        indices = []
        for stock_index, prices in enumerate(self.prices):
            # 逐只股票检查窗口；滑动视图不复制底层价格，临时掩码也只覆盖一只股票。
            windows = np.lib.stride_tricks.sliding_window_view(prices, pattern_length)
            valid = np.isfinite(windows).all(axis=1)
            valid &= windows.max(axis=1) > windows.min(axis=1)
            starts = np.flatnonzero(valid).astype(np.int32)
            indices.append(np.column_stack((np.full(len(starts), stock_index, dtype=np.int32), starts)))
        self.indices = np.concatenate(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        stock_index, start = self.indices[index]
        window = torch.tensor(self.prices[stock_index, start:start + self.pattern_length])
        window = window - window.min()
        return window / window.max()


def train_model(data, latent_dim=2, batch_size=1024, pattern_length=60, split_date="20140601", epochs=100, learning_rate=0.001, seed=42):
    """按窗口结束日期划分样本，训练一个指定维度的模型，返回最终模型和损失记录。"""
    if epochs < 1:
        raise ValueError("epochs 必须至少为 1")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PriceWindowDataset(data, pattern_length)
    split_days = data.loc[data["date"] >= split_date, "trade_day"]
    split_day = int(split_days.min()) if len(split_days) else int(data["trade_day"].max()) + 1
    end_days = dataset.indices[:, 1] + dataset.first_trade_day + pattern_length - 1
    train_indices = np.flatnonzero(end_days < split_day)
    val_indices = np.flatnonzero(end_days >= split_day)
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("筛选后的训练集或验证集为空，请检查日期范围、窗口长度和停牌记录")

    # 两个 Subset 共用价格矩阵；不启动额外 worker，避免复制数据集占用内存。
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, num_workers=0)
    model = AutoEncoder(pattern_length, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    print(f"device: {device}, train windows: {len(train_indices)}, validation windows: {len(val_indices)}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for windows in train_loader:
            windows = windows.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(windows), windows)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(windows)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for windows in val_loader:
                windows = windows.to(device)
                val_loss += F.mse_loss(model(windows), windows).item() * len(windows)

        train_loss /= len(train_indices)
        val_loss /= len(val_indices)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch + 1}/{epochs}, train MSE: {train_loss:.6f}, validation MSE: {val_loss:.6f}")

    return model.cpu(), pd.DataFrame(history)


def main():
    """正式训练入口；开发和测试时仅用小样本直接调用 train_model。"""
    with open("config.json") as file:
        config = json.load(file)
    with open("config_local.json") as file:
        config_local = json.load(file)

    pattern_length = 60
    latent_dim = 2
    data = pd.read_parquet(os.path.join(config_local["root_dir"], config["stock_daily_file"]), columns=["date", "code", "trade_day", "close", "is_suspended"], filters=[("date", ">=", "20100101"), ("date", "<=", "20151231")])
    model, history = train_model(data, latent_dim=latent_dim, pattern_length=pattern_length, batch_size=1024, epochs=100)
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "price_compression")
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"autoencoder_len{pattern_length}_dim{latent_dim}"
    torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}.pt"))
    history.to_csv(os.path.join(model_dir, f"{model_name}_loss.csv"), index=False)


if __name__ == "__main__":
    main()
