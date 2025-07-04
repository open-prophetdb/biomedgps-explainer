### 🎯 **任务目标**

我需要你帮助我将以下代码文件：

* `drugs4disease.py`
* `visualize.ipynb`
* `run.ipynb`

封装成一个完整、可安装、可复用的 Python 包，用于围绕任何指定疾病生成潜在药物列表、支持用户自定义复杂筛选条件、并输出图表。

---

### 📌 **功能要求**

#### 1️⃣ **主功能**

* 接收用户输入的疾病名称（或模型支持的疾病 ID）。
* 调用已有逻辑生成 Potential Drug List（`annotated_drugs.xlsx`）。
* 支持用户通过逻辑表达式对结果进行筛选，例如：

  ```
  column_a_name >= 0.5 && column_b_name != "unknown"
  ```

  要求支持常见的操作符（`==`, `!=`, `<`, `>`, `<=`, `>=`）与逻辑运算符（`&&`, `||`），并对表达式解析后应用于 DataFrame。
* 输出筛选后的 `final_drugs.xlsx`。
* 提供单独的模块用于可视化，生成常见图表（例如柱状图、散点图、热图），保存为文件（PNG 或 PDF）。

---

#### 2️⃣ **代码结构**

请将功能分为两个主要子模块：

```
drugs4disease/
  __init__.py
  core.py        # 数据生成与处理逻辑
  filter.py      # 表达式解析与筛选逻辑
  visualizer.py  # 可视化逻辑（与运行解耦）
  cli.py         # 命令行接口
tests/
  test_core.py
  test_filter.py
  test_visualizer.py
```

* `core.py`: 包含潜在药物列表生成逻辑。
* `filter.py`: 实现逻辑表达式解析并对 DataFrame 进行筛选。注意，新生成的表既需要包含原始数据表，也需要包含筛选后的数据表。
* `visualizer.py`: 提供单独的可视化功能，可被独立调用。除了生成图，还需要包含一系列解读的文字。放入Report.md中（visualize.ipynb已经为每个图准备了相应文字，请以此为参考），引用图表。
* `cli.py`: 提供 CLI 接口，支持：

  * 生成数据：`drugs4disease run --disease "Breast Cancer" --output-dir ./results`
  * 筛选数据：`drugs4disease filter --expression 'score >= 0.5 && status != "unknown"' --input ./results/annotated_drugs.xlsx --output ./results/final_drugs.xlsx`
  * 生成图表：`drugs4disease visualize --input ./results/final_drugs.xlsx --output-dir ./results/figures`

---

#### 3️⃣ **CLI 特性**

* 使用 `click` 构建 CLI。
* 支持将筛选表达式从命令行传入。

---

#### 4️⃣ **包与文档**

* 提供 `setup.py` 或 `pyproject.toml`。
* 提供完整 `README.md`：

  * 安装方法
  * 示例 CLI 命令
  * 参数说明
  * 示例输出截图或文件结构

---

### ⚙️ **开发细节**

✅ 筛选表达式解析可用 `pandas.query` 或自定义解析器（支持逻辑运算符转换：`&&` ➡️ `and`, `||` ➡️ `or`）。

✅ 图表生成推荐支持：

* 柱状图（药物分数分布）
* 热图（药物-指标矩阵）
* 散点图（例如分数 vs 另一个指标）

✅ 单元测试覆盖筛选表达式解析、数据生成主流程、图表文件生成。

✅ Python 3.8+。

---

### 🚀 **输出要求**

请生成：

* 包目录结构说明。
* 各模块文件的主要代码（重点是表达式解析、CLI、可视化入口）。
* 安装与使用说明。
* 示例 CLI 调用。
* 示例输出文件清单。

---

### 📝 **上下文补充**

> 现有文件（`drugs4disease.py`, `visualize.ipynb`, `run.ipynb`）的核心逻辑可以直接集成，无需重写算法，只需合理封装。
