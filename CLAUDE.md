# CLAUDE.md

A 股因子研究项目：整理分钟/日线行情，构建高频与中频因子，做单因子和批量因子评估。项目说明、配置字段和输出文件的详细含义见 [README.md](README.md)，本文件只记录写代码时必须遵守的规范。

## 运行方式

在 [main.py](main.py) 中修改 `CONFIG_FILE` 选择任务，然后执行 `python main.py`。本地数据根目录由 `config_local.json` 的 `root_dir` 指定，配置中以 `_dir` / `_path` 结尾的字段会自动拼到该根目录下。

| 任务 | 配置文件 | 入口类 |
| --- | --- | --- |
| 数据更新 | `config/config_data_update.json` | `DataUpdater` |
| 因子构建 | `config/config_factor_construction.json` | `HighFreqFactorConstructor` / `MiddleFreqFactorConstructor` |
| 因子评估 | `config/config_factor_evaluation.json` | `SingleFactorEvaluation` / `MultiFactorEvaluation` |

Python 环境：自动使用 `~/.venvs/research/` 中的虚拟环境，运行任何 Python 命令时都用 `~/.venvs/research/bin/python`（例如 `~/.venvs/research/bin/python main.py`），不要用系统 Python。

项目没有测试套件，改动后通过实际运行对应任务来验证。

## 代码书写规范

以下四条规范适用于本项目的所有代码改动，优先级高于通用的 Python 风格习惯（例如不需要为了 79 字符行宽而折行）。

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
- 新增配置项写进对应的 `config/*.json`，并在相关类的 `__init__` 中与其他字段一样读取。

项目中已有的扩展方式，新增内容时直接照做：

- **新增高频因子**：在 [factors/construction/high_freq_factor_calculation/](factors/construction/high_freq_factor_calculation/) 下新建与类名同名的文件，继承 `_BaseHighFreqFactor`，实现 `calculate()`，返回 `code`、`date` 和以 `factor_name` 命名的因子列。
- **新增中频因子**：在 [factors/construction/middle_freq_factor_calculation/](factors/construction/middle_freq_factor_calculation/) 下同样处理，继承 `_BaseMiddleFreqFactor`，并设置 `lookback_days`；不要修改共享的 `self.daily`。
- 因子类需带上 `factor_name`、`factor_type`、`calculation_logic` 等类属性，写法参照同目录下的现有因子。
- 因子文件名即类名，构建器按文件名自动注册，不需要改动构建器代码。

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

## 数据约定

- 日期统一为 `YYYYMMDD` 字符串，股票代码为六位字符串；读 CSV 时用 `dtype={"code": str, "date": str}` 保留前导零。
- DataFrame 以 `code`、`date` 为主键列，因子文件每个因子一个 parquet，因子值列名与因子类名一致。
- 注释和 docstring 使用中文，写法与周围代码保持一致。
