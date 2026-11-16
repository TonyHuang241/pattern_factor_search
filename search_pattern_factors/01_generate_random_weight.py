import glob
import json
import os

import numpy as np
import pandas as pd

if __name__ == "__main__":
    config = json.load(open("config/config_pattern_search.json", encoding="utf-8"))
    config_local = json.load(open("config_local.json", encoding="utf-8"))
    config = {key: os.path.join(config_local["root_dir"], value) if key.endswith(("_dir", "_file")) else value for key, value in config.items()}
    os.makedirs(config["random_weight_dir"], exist_ok=True)
    # 清空上次生成的权重文件，避免旧配置的结果被后续步骤一起处理
    for file in glob.glob(os.path.join(config["random_weight_dir"], "random_weight_*.parquet")):
        os.remove(file)

    # 多项式个数与窗口长度的每种组合各生成一个权重文件
    for num_polynomials in config["num_polynomials"]:
        for window_length in config["window_length"]:
            # 第 1 到第 n 阶 Legendre 多项式在窗口内等距点上的取值，形状为 (window_length, n)
            basis = np.polynomial.legendre.legvander(np.linspace(-1, 1, window_length), num_polynomials)[:, 1:]

            # n 维单位球体内均匀取点：标准正态取方向，U^(1/n) 取半径；每个组合都用同一个 seed 初始化
            rng = np.random.default_rng(config["seed"])
            coordinate = rng.standard_normal((config["num_weights"], num_polynomials))
            coordinate = coordinate / np.linalg.norm(coordinate, axis=1, keepdims=True) * rng.uniform(size=(config["num_weights"], 1)) ** (1 / num_polynomials)

            # 坐标与基线性组合得到权重，w_0 对应窗口内最早的一天，减去自身均值使权重和为 0，再除以绝对值之和
            weight = pd.DataFrame(coordinate @ basis.T, columns=[f"w_{j}" for j in range(window_length)])
            weight = weight.sub(weight.mean(axis=1), axis=0)
            weight = weight.div(weight.abs().sum(axis=1), axis=0)
            weight.insert(0, "weight_id", range(config["num_weights"]))
            weight.to_parquet(os.path.join(config["random_weight_dir"], f"random_weight_n{num_polynomials}_len{window_length}.parquet"), index=False)
            print(f"已生成 {config['num_weights']} 条权重，n = {num_polynomials}，窗口 {window_length} 天")
