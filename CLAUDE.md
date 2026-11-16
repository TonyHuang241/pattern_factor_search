# CLAUDE.md

A 股随机权重因子搜索项目：用第 1 到第 n 阶 Legendre 多项式生成随机权重（均值为 0、绝对值之和为 1），把过去窗口期内每天的日度股票回报升维成 6 维（回报、绝对值、符号、平方、三次方、过去 `rolling_mean_window` 日滚动平均），各维度在窗口内做 zscore 后分别与权重加权求和，6 个结果取平均得到因子值，再评估因子的 RankIC 和 RankICIR。本文件只记录写代码时必须遵守的规范。

## 运行方式

三个脚本共用 `config/config_pattern_search.json`，在项目根目录下按顺序执行。本地数据根目录由 `config_local.json` 的 `root_dir` 指定，配置中以 `_dir` / `_file` 结尾的字段会拼到该根目录下。

| 步骤 | 脚本 | 输出 |
| --- | --- | --- |
| 1. 生成随机权重 | `search_pattern_factors/01_generate_random_weight.py` | `random_weight/random_weight_n8_len60.parquet` |
| 2. 计算因子值和样本内 RankIC | `search_pattern_factors/02_calculate_factor_value.py` | `factor_value/factor_ic_n8_len60.parquet`、`factor_value/factor_value_n8_len60.parquet` |
| 3. 样本外评估和出图 | `search_pattern_factors/03_evaluate_factor.py` | `output/factor_evaluation_n8_len60.html` |

文件名中的 `n8_len60` 表示 Legendre 多项式个数为 8、窗口长度为 60。配置里的 `num_polynomials` 和 `window_length` 都是 list，第 1 步按两者的笛卡尔积每个组合生成一个权重文件；第 2、3 步自动处理上一步输出文件夹中的所有文件，有几个文件就算几组，窗口长度以各文件自身为准。因子太多，全部因子值不落盘：第 2 步边算因子边算样本内 RankIC，挑选 RankIC 绝对值最大的 `num_top_factors` 个因子，只保存入选因子的具体值。

Python 环境：自动使用 `~/.venvs/research/` 中的虚拟环境，运行任何 Python 命令时都用 `~/.venvs/research/bin/python`，不要用系统 Python。Windows 本机没有这个环境，改用 `C:/Users/hung/projects/.venv/Scripts/python.exe`。

项目没有测试套件，改动后通过实际运行对应脚本来验证。

## 代码书写规范

以下六条规范适用于本项目的所有代码改动，优先级高于通用的 Python 风格习惯（例如不需要为了 79 字符行宽而折行）。

### 1. 一行能写完就不分行，语义上是两步就拆成两行

判断标准是语义而不是长度：一个完整的操作写在一行，不要为了行宽把参数逐个换行；但如果一行里串了两个可以分别说清楚的步骤，就拆开，避免又臭又长的链式调用。

```python
# 推荐：一个操作一行，即使这一行比较长
constructor_class = {"high": HighFreqFactorConstructor, "middle": MiddleFreqFactorConstructor}[config.get("construction_mode", "high")]
stock_st_status = pd.read_parquet(os.path.join(self.stock_st_status_dir, "stock_st_status.parquet"), columns=["code", "date", "is_st"])

# 不推荐：一个操作被无意义地拆成多行
stock_st_status = pd.read_parquet(
    os.path.join(self.stock_st_status_dir, "stock_st_status.parquet"),
    columns=["code", "date", "is_st"],
)
```

```python
# 推荐：筛选、合并是两个步骤，分两行
factor = factor.loc[factor["date"].isin(pending_dates)]
factor = factor.merge(stock_pool, on=["code", "date"], how="inner")

# 不推荐：多个步骤挤在一行
factor = factor.loc[factor["date"].isin(pending_dates)].merge(stock_pool, on=["code", "date"], how="inner").drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
```

### 2. 改动尽可能小，嵌入现有框架

- 只改完成需求所必需的行，不顺手重构、重命名、调整格式或改动无关代码。
- 新功能优先放进已有的类和方法里，沿用已有的命名、配置读取方式和数据流，不另起一套结构。
- 不为一次性逻辑新建文件、工具模块、基类或辅助函数；确实需要新方法时，加在相关的现有类中。
- 新增配置项写进 `config/config_pattern_search.json`，在脚本中与其他字段一样从 `config` 读取。

### 3. 用简单函数，少用高级函数

优先使用最基础、最直白的写法：普通的 `for` 循环、`if` 判断和 pandas 的基本列运算。除非简单写法明显做不到或性能上不可接受，否则不要使用下面这类写法：

- `functools.reduce`、`itertools`、`map` / `filter` 配 `lambda`
- 装饰器、元类、生成器、上下文管理器等自定义的语言高级特性
- `df.pipe`、`df.eval`、`df.query`、`apply` 配复杂 `lambda`、多层嵌套的推导式
- 为了简短而牺牲可读性的技巧性写法

```python
# 推荐
for window in rolling_windows:
    factor[f"{factor_name}_{window}_m"] = factor_by_code.rolling(window).mean().droplevel(0).reindex(factor.index)

# 不推荐
factor = reduce(lambda df, w: df.assign(**{f"{factor_name}_{w}_m": lambda x: x.groupby("code")[factor_name].transform(lambda s: s.rolling(w).mean())}), rolling_windows, factor)
```

### 4. 少建新变量，能用 DataFrame 列操作就用列操作

中间结果直接写成 DataFrame 的列，或者覆盖原来的 DataFrame 变量，不要拆出一堆独立的 Series、数组或临时 DataFrame。

```python
# 推荐：中间量作为列留在同一个 DataFrame 里
daily["trade_day"] = daily["date"].rank(method="dense").astype(int)
daily["previous_trade_day"] = daily["trade_day"] - self.lookback_days
daily[self.factor_name] = daily["close"] / daily["previous_close"] - 1

# 不推荐：每个中间量一个新变量
trade_day = daily["date"].rank(method="dense").astype(int)
previous_trade_day = trade_day - self.lookback_days
close = daily["close"]
previous_close = daily["previous_close"]
factor_value = close / previous_close - 1
```

```python
# 推荐：逐步处理时复用同一个变量名
factor = factor.loc[factor["date"].isin(pending_dates)]
factor = factor.merge(stock_pool, on=["code", "date"], how="inner")

# 不推荐：每一步都换一个新名字
factor_filtered = factor.loc[factor["date"].isin(pending_dates)]
factor_merged = factor_filtered.merge(stock_pool, on=["code", "date"], how="inner")
```

只有当一个值会被多处使用、或者不放进变量就无法表达时，才新建变量。

### 5. 代码简洁可读，注释要短

- 代码兼顾简洁和可读性，优先使用 numpy / pandas 中最常用的函数，不用复杂、冷门的 pandas 高级函数。
- 注释能一句话说完就只写一句，不重复代码本身已经表达的内容。

### 6. 注意计算效率，充分利用多进程

- 本机有 72 个逻辑线程和 128G 内存，耗时的计算用多进程并行，充分利用计算资源。
- 大矩阵运算用 numpy 向量化完成，不逐行逐股票循环；同时控制总内存占用，不超过物理内存。
- Windows 下单个进程池最多 61 个进程，多进程入口必须放在 `if __name__ == "__main__":` 之下。

## 数据约定

- 日期统一为 `YYYYMMDD` 字符串，股票代码为六位字符串；读 CSV 时用 `dtype={"code": str, "date": str}` 保留前导零。
- DataFrame 以 `code`、`date` 为主键列；日线数据的 `close` 已是后复权价，相邻两日之比即真实收益。
- 因子值在当日收盘后得到，评估用的未来收益从下一日 10 点开盘价算到其后第 `forward_period` 个交易日的 10 点开盘价；10 点价格来自 `stock_10am_file`（未复权），乘日线的 `adj_factor` 得到后复权价。
- 第 3 步对入选因子分样本内、样本外各算一套指标：IC、RankIC、两者的 t 值，以及等分 `num_groups` 组后各组区间累计收益与组号的 Pearson 相关系数（单调性）。
- 输出文件名统一带 `n{多项式个数}_len{窗口长度}`；`random_weight_*.parquet` 每行一条权重，`w_0` 对应窗口内最早一天；`factor_ic_*.parquet` 每行一个随机权重的样本内 RankIC 和 RankICIR（列名 `rank_ic`、`rank_icir`）；`factor_value_*.parquet` 每行一个 `code`、`date`，只含入选因子的因子列 `factor_{weight_id}`。
- 股票池为 `stock_pool_index_code` 指定指数的成分股（默认中小300），每个交易日取不晚于当日的最近一期成分股；RankIC 和分层都只在股票池内计算，`factor_value_*.parquet` 只含当日在股票池内的股票。
- 注释和 docstring 使用中文，写法与周围代码保持一致。
