import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        # 对原始价格窗口做 Min-Max 标准化，压缩为单位长度向量后解码还原；同时返回标准化窗口作为重建目标。
        x = x - x.min(dim=-1, keepdim=True).values
        x = x / x.max(dim=-1, keepdim=True).values
        return x, self.decoder(F.normalize(self.encoder(x), p=2, dim=-1))


def plot_loss(history, config):
    """绘制训练集和验证集的双对数损失曲线，返回图片对象。"""
    fig, ax = plt.subplots()
    ax.loglog(history["epoch"], history["train_loss"], label="Train")
    ax.loglog(history["epoch"], history["val_loss"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE", title=f"latent_dim={config['latent_dim']}, pattern_length={config['pattern_length']}")
    ax.legend()
    return fig


def train_model(data, config):
    """将收盘价宽表展开为三维滑动窗口，按窗口结束日期划分样本，按 config 中的参数训练并早停，返回验证损失最优的模型和损失曲线图片。"""
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_day = int(data.loc[data["date"] >= config["split_date"], "trade_day"].min())

    # 宽表行为交易日、列为股票；停牌日不进宽表，补齐交易日后即为 NaN。
    data = data.loc[~data["is_suspended"]]
    data = data.pivot(index="trade_day", columns="code", values="close")

    # 三维视图 (窗口数, pattern_length, 股票数) 不复制数据；每个窗口剔除期内有缺失或无波动的股票。
    windows = np.lib.stride_tricks.sliding_window_view(data.to_numpy(dtype=np.float32), config["pattern_length"], axis=0)
    windows = windows.transpose(0, 2, 1)
    valid = np.isfinite(windows).all(axis=1) & (windows.max(axis=1) > windows.min(axis=1))
    window_index, stock_index = np.nonzero(valid)
    end_days = window_index + int(data.index.min()) + config["pattern_length"] - 1

    # 按 (窗口, 股票) 取出保留的价格序列，复制为 (样本数, pattern_length) 的二维数组；Min-Max 标准化在模型 forward 中完成。
    windows = windows[window_index, :, stock_index]
    train_windows = torch.from_numpy(windows[end_days < split_day]).to(device)
    val_windows = torch.from_numpy(windows[end_days >= split_day]).to(device)

    model = AutoEncoder(config["pattern_length"], config["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    print(f"device: {device}, train windows: {len(train_windows)}, validation windows: {len(val_windows)}")

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in torch.randperm(len(train_windows), device=device).split(config["batch_size"]):
            windows = train_windows[batch]
            optimizer.zero_grad(set_to_none=True)
            windows, reconstruction = model(windows)
            loss = F.mse_loss(reconstruction, windows)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(windows)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for windows in val_windows.split(config["batch_size"]):
                windows, reconstruction = model(windows)
                val_loss += F.mse_loss(reconstruction, windows).item() * len(windows)

        train_loss /= len(train_windows)
        val_loss /= len(val_windows)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if epoch%50 == 49:
            print(f"epoch {epoch + 1}/{config['epochs']}, train MSE: {train_loss:.6f}, validation MSE: {val_loss:.6f}")

        # 早停：验证损失创新低时保存参数，连续 patience 个 epoch 未创新低则停止训练。
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config["patience"]:
            print(f"early stop at epoch {epoch + 1}, best epoch: {best_epoch + 1}, best validation MSE: {best_val_loss:.6f}")
            break
    else:
        print(f"training finished, best epoch: {best_epoch + 1}, best validation MSE: {best_val_loss:.6f}")

    model.load_state_dict(best_state)
    return model.cpu(), plot_loss(pd.DataFrame(history), config)


if __name__ == "__main__":
    # 正式训练入口；开发和测试时仅用小样本直接调用 train_model。
    config = json.load(open("config/config_autoencoder.json"))
    config_local = json.load(open("config_local.json"))
    output_dir = os.path.join(config_local["root_dir"], config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(config_local["root_dir"], config["model_path"]), exist_ok=True)

    data = pd.read_parquet(os.path.join(config_local["root_dir"], config["stock_daily_file"]), filters=[("date", ">=", config["start_date"]), ("date", "<=", config["end_date"])])
    for i, latent_dim in enumerate([2, 4, 8, 16]):
        for j, pattern_length in enumerate([20, 60, 120, 250]):
            print(f"[{i * 4 + j + 1}/16] latent_dim={latent_dim}, pattern_length={pattern_length}", flush=True)
            config["latent_dim"] = latent_dim
            config["pattern_length"] = pattern_length
            model, fig = train_model(data, config)

            # 模型和损失曲线分别保存，文件名标明窗口长度和压缩维度。
            filename = f"autoencoder_len{pattern_length}_dim{latent_dim}"
            model_path = os.path.join(config_local["root_dir"], config["model_path"], filename)
            torch.save(model.state_dict(), f"{model_path}.pt")
            fig.savefig(os.path.join(output_dir, f"{filename}_loss.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
