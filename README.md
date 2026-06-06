# 零售销售额的预测与数据分析

> 2026年第五届全国大学生数据统计与分析竞赛 — 题目B

---

## 项目文件结构

```
├── 数据文件
│   ├── data.csv                 原始数据 (9800行，18列)
│   ├── data_cleaned.csv         清洗后数据 (9800行，18列)
│   ├── feature_matrix.csv       数值化特征矩阵 — 明细行级别 (9800行×43列)
│   └── feature_order.csv        数值化特征矩阵 — 订单级别   (4921行×24列)
│
├── 脚本文件 (按执行顺序)
│   ├── clean_data.py            数据清洗脚本
│   ├── look.py                  特征数值化脚本
│   ├── temp.py                  临时测试脚本（可忽略）
│   │
│   └── src/                     问题分析脚本
│       ├── data.py              数据概览（基本信息、缺失值、描述统计）
│       ├── problem1.py          问题1：基本统计量与多维分布分析
│       ├── problem2.py          问题2：影响因素的方差分析与相关性分析
│       └── cyclical_analysis.py 周期性/季节性分析（ACF/PACF、季节分解）
│
├── 说明文档
│   ├── README.md                本文件
│   ├── cleaning_notes.txt       数据清洗说明（清洗了什么、为什么）
│   ├── docs/data.txt            数据质量问题汇总
│   │
│   └── ouput/                   分析结果
│       ├── problem1.txt         问题1文本输出
│       ├── problem1_trend.png   日/月/年销售额趋势图
│       ├── problem1_distribution.png  多维度分布图
│       ├── problem2_analysis.md 问题2分析报告
│       ├── problem2_correlation.png   相关性可视化图
│       ├── model_analysis.md    建模方案分析（时间序列 vs 回归）
│       ├── cycle_acf_pacf.png   自相关/偏自相关图
│       ├── cycle_dayofweek.png  星期几效应图
│       ├── cycle_growth.png     增长趋势图
│       ├── cycle_seasonal_index.png  季节指数图
│       └── cycle_year_overlay.png    逐年叠加对比图
│
└── 题目文件
    └── 题目B：零售销售额的预测与数据分析.pdf  竞赛原题
```

## 数据字段说明

`data.csv` / `data_cleaned.csv` 共18列：

| 字段 | 类型 | 说明 |
|------|------|------|
| Row ID | 整数 | 行号 |
| Order ID | 文本 | 订单编号 (如 CA-2024-152156) |
| Order Date | 日期 | 下单日期 (dd/mm/yyyy) |
| Ship Date | 日期 | 发货日期 (dd/mm/yyyy) |
| Ship Mode | 文本 | 发货方式 (First/Second/Standard Class / Same Day) |
| Customer ID | 文本 | 客户编号 |
| Customer Name | 文本 | 客户姓名 |
| Segment | 文本 | 客户细分 (Consumer / Corporate / Home Office) |
| Country | 文本 | 国家 (全部为 United States) |
| City | 文本 | 城市 |
| State | 文本 | 州 |
| Postal Code | 文本 | 邮编 |
| Region | 文本 | 区域 (East / West / Central / South) |
| Product ID | 文本 | 产品编号 |
| Category | 文本 | 产品大类 (Furniture / Office Supplies / Technology) |
| Sub-Category | 文本 | 产品子类 (17种，如 Chairs, Binders, Phones 等) |
| Product Name | 文本 | 产品名称 |
| **Sales** | 小数 | **销售额（预测目标）** |

## 各文件详细说明

### 数据文件

#### `data.csv` — 原始数据
比赛提供的原始数据集，含少量质量问题（邮编缺失、发货日期年份错误、邮编格式不统一等）。共9800条订单明细。

#### `data_cleaned.csv` — 清洗后数据
运行 `clean_data.py` 生成，对原始数据做了3项修复：
1. **42条发货日期年份错误**：Ship Date 年份 2019 → 2026（订单在2025年底，发货在2026年初）
2. **11条缺失邮编**：Burlington, Vermont → 05401
3. **429条4位邮编**：补零为5位（如 2908 → 02908）

#### `feature_matrix.csv` — 明细行级别数值化特征
运行 `look.py` 生成，将 `data_cleaned.csv` 全部转为纯数值（43列），**每行是一件商品**。适合预测单品销售额。包括：
- 日期拆解为年/月/日/星期/季度/是否周末/是否旺季
- 类别列做独热编码（Ship Mode / Segment / Region / Category）或标签编码（City / State / Sub-Category）
- 从ID列提取客户统计特征（历史总消费、订单频次、平均消费额）和产品统计特征（历史销量、平均单价）
- 目标变量：`Sales`（单件商品销售额）和 `LogSales`（对数变换后的销售额）

#### `feature_order.csv` — 订单级别数值化特征
运行 `look.py` 生成，**每行是一个订单**。将同一订单的多件商品合并：
- 聚合得到：`TotalSales`（订单总销售额 ← 预测目标）、`ItemCount`（商品件数）、`AvgItemPrice`（均价）
- 保留订单维度的类别特征和日期特征
- 适合预测订单总金额

### 脚本文件

#### `clean_data.py` — 数据清洗
读取 `data.csv` → 修复日期/邮编问题 → 输出 `data_cleaned.csv`

#### `look.py` — 特征数值化
读取 `data_cleaned.csv` → 将文本/日期转为数值特征 → 输出 `feature_matrix.csv` 和 `feature_order.csv`。适合初学者阅读，每步有详细中文注释。

#### `src/data.py` — 数据概览
运行 `python src/data.py`，打印数据的基本信息：行数列数、每列数据类型、缺失值统计、数值列描述统计、类别列取值分布。

#### `src/problem1.py` — 基本统计与可视化
运行 `python src/problem1.py`，输出：
- 销售额的基本统计量（均值、中位数、标准差等）
- 总销售额、总订单数、客户数等业务指标
- 生成 `ouput/problem1_trend.png`（日/月/年趋势图）
- 生成 `ouput/problem1_distribution.png`（按年份/季度/月份/区域/品类/客户细分的销售分布）
- 结果同时打印到控制台并保存至 `ouput/problem1.txt`

#### `src/problem2.py` — 影响因素分析
运行 `python src/problem2.py`，通过 ANOVA 方差分析计算各因素对销售额的影响大小（η²效应量），识别关键影响因素：
- 明细行级：Category > Sub-Category > Region > Ship Mode ...
- 订单级：ShipMode > Segment > Region ...
- 客户级：Segment > Region
- 生成 `ouput/problem2_correlation.png`（相关性与效应量可视化）
- 输出报告保存至 `ouput/problem2_analysis.md`

#### `src/cyclical_analysis.py` — 周期性分析
运行 `python src/cyclical_analysis.py`，对销售额进行时间序列分析：
- 自相关（ACF）和偏自相关（PACF）图
- 季节分解（趋势、季节、残差）
- 星期几效应、年度增长趋势、季节指数
- 生成多张图表保存至 `ouput/`

---

## 快速开始

```bash
# 1. 数据清洗
python clean_data.py

# 2. 特征数值化
python look.py

# 3. 数据分析
python src/data.py
python src/problem1.py
python src/problem2.py
python src/cyclical_analysis.py

# 4. 建模 (以 feature_matrix.csv 为例)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("feature_matrix.csv")
X = df.drop(columns=["Sales", "Order Date", "Ship Date"])
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

---

## 分析结果汇总

所有图表和分析报告位于 `ouput/` 目录，包括：
- **问题1**：销售额基本特征与多维度分布
- **问题2**：各因素对销售额的统计显著性检验
- **周期性分析**：时间序列的季节性与趋势特征
- **建模方案**：时间序列预测 vs 回归预测的选型建议

详细分析报告请查看 `ouput/model_analysis.md`。
