"""
数据数值化转换脚本
======================
用途: 将清洗后的 data_cleaned.csv 转化为纯数值特征矩阵，
     供数学建模（回归/分类/时间序列等）直接使用。

输出文件:
  - feature_matrix.csv : 每个订单明细行一条记录，含所有数值特征
  - feature_order.csv  : 按订单聚合，每订单一条记录

初学者友好提示:
  - 每一步都有详细注释说明"为什么这样做"
  - 运行方式: python look.py
  - 依赖: pip install pandas numpy
"""

import pandas as pd
import numpy as np

# =============================================================
# 第一步: 读取清洗后的数据
# =============================================================
print("=" * 60)
print("第一步: 读取数据")
print("=" * 60)

df = pd.read_csv("data_cleaned.csv", encoding="gbk")

# 显示列名，让初学者知道有哪些数据可用
print(f"\n数据大小: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"\n列名列表:\n{list(df.columns)}")

# =============================================================
# 第二步: 处理日期列 — 把"日期"变成"数字"
# =============================================================
# 为什么要转? 模型不认识 "08/11/2024"，只认识数字。
# 我们从日期中提取: 年、月、日、星期几、季度、是否周末、是否旺季
print("\n" + "=" * 60)
print("第二步: 日期 → 数值特征")
print("=" * 60)

# 先把字符串转成 pandas 的 datetime 类型
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")

# --- 从 Order Date 中提取时间特征 ---
df["OrderYear"] = df["Order Date"].dt.year          # 年份: 2022~2025
df["OrderMonth"] = df["Order Date"].dt.month         # 月份: 1~12
df["OrderDay"] = df["Order Date"].dt.day             # 日: 1~31
df["OrderDayOfWeek"] = df["Order Date"].dt.dayofweek # 星期几: 0=周一, 6=周日
df["OrderQuarter"] = df["Order Date"].dt.quarter     # 季度: 1~4

# 是否周末 (周末=1, 工作日=0) — 周末购物行为可能不同
df["IsWeekend"] = (df["OrderDayOfWeek"] >= 5).astype(int)

# 是否年末旺季 (11月~12月) — 黑五/圣诞促销季销售额通常更高
df["IsHolidaySeason"] = df["OrderMonth"].isin([11, 12]).astype(int)

# --- 从 Ship Date 提取特征 ---
df["ShipYear"] = df["Ship Date"].dt.year
df["ShipMonth"] = df["Ship Date"].dt.month
df["ShipDay"] = df["Ship Date"].dt.day

# --- 计算发货耗时 (天) — 物流速度影响客户体验 ---
df["ShippingDays"] = (df["Ship Date"] - df["Order Date"]).dt.days

print("新增日期特征: OrderYear, OrderMonth, OrderDay, OrderDayOfWeek, \n"
      "             OrderQuarter, IsWeekend, IsHolidaySeason, \n"
      "             ShipYear, ShipMonth, ShipDay, ShippingDays")

# =============================================================
# 第三步: 处理类别列 — 把"文字"变成"数字"
# =============================================================
# 模型不认识 "Consumer"、"Furniture" 这些词，需要转为数字。
# 有两种转换方式:
#   ① 独热编码 (One-Hot): 类别少时用，如 Region 只有4个取值
#   ② 标签编码 (Label): 类别多时用，如 City 有几百个
print("\n" + "=" * 60)
print("第三步: 类别文字 → 数值")
print("=" * 60)

# ---- 3a) 低基数类别 → 独热编码 (One-Hot Encoding) ----
# 这些列取值很少(≤10种)，独热编码后每个取值变成一列 0/1
low_card_cols = ["Ship Mode", "Segment", "Region", "Category", "IsHolidaySeason"]
print(f"\n低基数类别 (直接独热编码): {low_card_cols}")

for col in low_card_cols:
    # get_dummies 会把 "Ship Mode" 变成 "Ship Mode_Standard Class" 等列
    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    # 删除原始文字列，因为模型不能用文字
    df.drop(columns=[col], inplace=True)

# ---- 3b) 高基数类别 → 标签编码 (Label Encoding) ----
# City有几百个城市，State有50个州，Sub-Category有十几个
# 标签编码: 每个不同的文字→一个数字(0,1,2,...)
from sklearn.preprocessing import LabelEncoder

label_cols = ["City", "State", "Sub-Category"]
print(f"高基数类别 (标签编码): {label_cols}")

for col in label_cols:
    le = LabelEncoder()
    df[col + "_Encoded"] = le.fit_transform(df[col].astype(str))
    # 保留编码后的数字列，删除原始文字列
    df.drop(columns=[col], inplace=True)
    # 打印编码映射，方便理解
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col}: {len(mapping)} 种取值, 例如: {dict(list(mapping.items())[:3])}...")

# ---- 3c) 处理 Ship Mode 和 Segment 的标签编码问题 ----
# 注意: 我们已经在上面用 get_dummies 处理过 Ship Mode, Segment 了
# 但是之前代码里 low_card_cols 已经包括了它们，所以无冲突

# Country 只有 "United States" 一个值，没有区分度，直接删除
df.drop(columns=["Country"], inplace=True)
print("\n删除无信息列: Country (只有 United States 一个值)")

# =============================================================
# 第四步: 处理 ID 类列
# =============================================================
# 纯编号 (Row ID, Order ID, Product ID, Customer ID) 本身是随机标识，
# 直接作为数字输入模型会导致过拟合。但我们可提取其中的有用信息。

print("\n" + "=" * 60)
print("第四步: ID 列处理")
print("=" * 60)

# Row ID 仅仅是行号，无预测价值，删除
df.drop(columns=["Row ID"], inplace=True)
print("删除 Row ID (纯行号, 无意义)")

# Order ID 如 "CA-2024-152156", 中间的 2024 是年份信息
# 提取 Order 中的年份前缀，作为特征 (可能与订单量趋势有关)
df["OrderYearPrefix"] = df["Order ID"].str.extract(r"-(20\d{2})-").astype(int)
# 还可以统计该 Order ID 出现了几次 (即该订单包含几件商品)
order_size = df.groupby("Order ID")["Sales"].transform("count")
df["OrderSize"] = order_size
print("从 Order ID 中提取: OrderYearPrefix, OrderSize")

# Customer ID 如 "CG-12520"
# 提取客户编号的后半部分数字，作为客户特征
df["CustomerNum"] = df["Customer ID"].str.extract(r"-(\d+)").astype(int)
# 统计每个客户的总消费和订单数作为特征
customer_stats = df.groupby("Customer ID").agg(
    CustomerTotalSpent=("Sales", "sum"),
    CustomerOrderCount=("Order ID", "nunique"),
    CustomerAvgSpent=("Sales", "mean"),
)
df = df.merge(customer_stats, on="Customer ID", how="left")
print("从 Customer ID 中提取: CustomerNum, CustomerTotalSpent, \n"
      "                         CustomerOrderCount, CustomerAvgSpent")

# Product ID 如 "FUR-BO-10001798"
# 提取产品类别前缀 (FUR/OFF/TEC)
df["ProductCategory"] = df["Product ID"].str.extract(r"^([A-Z]+)-")
# 产品类别前缀也是文字，需要编码
le_prod = LabelEncoder()
df["ProductCategory_Encoded"] = le_prod.fit_transform(df["ProductCategory"].astype(str))
df.drop(columns=["ProductCategory"], inplace=True)
product_stats = df.groupby("Product ID").agg(
    ProductSalesCount=("Sales", "count"),
    ProductAvgPrice=("Sales", "mean"),
)
df = df.merge(product_stats, on="Product ID", how="left")
print("从 Product ID 中提取: ProductCategory_Encoded, ProductSalesCount, \n"
      "                         ProductAvgPrice")

# 删除原始 ID 列和姓名字段 (模型无法直接使用)
df.drop(columns=["Order ID", "Customer ID", "Product ID",
                  "Customer Name", "Product Name"], inplace=True)
print("删除原始 ID/姓名列: Order ID, Customer ID, Product ID, \n"
      "                      Customer Name, Product Name")

# Postal Code 在清洗后已是5位数字，可直接保留作为数值
# 但邮编本身是地理位置编码，直接用数值可能不合适，保留即可

# =============================================================
# 第五步: 处理目标变量 (Sales) — 可选的对数变换
# =============================================================
# Sales 是我们要预测的目标。原始数据右偏严重 (均值230 >> 中位数54)，
# 如果做回归模型，取对数可以让分布更接近正态，提升模型效果。

print("\n" + "=" * 60)
print("第五步: 目标变量处理")
print("=" * 60)

# 保留原始 Sales 作为目标
# 另外创建 log(Sales) 供后续建模选用
df["LogSales"] = np.log1p(df["Sales"])  # log1p = log(1+x), 避免 log(0) 问题

print("新增 LogSales = log(1+Sales)，适合右偏数据的回归建模")

# =============================================================
# 第六步: 最终检查 & 保存
# =============================================================
print("\n" + "=" * 60)
print("第六步: 输出结果")
print("=" * 60)

# 查看最终数值矩阵的信息
print(f"\n最终特征矩阵大小: {df.shape[0]} 行 × {df.shape[1]} 列")

# 查看所有列的类型 — 应该全部是数字 (int/float)
print(f"\n所有列的数据类型:")
for col in df.columns:
    print(f"  {col:40s}  {df[col].dtype}")

# 检查是否还有非数值列 (模型不能处理的)
non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
if non_numeric:
    print(f"\n⚠ 警告: 还有 {len(non_numeric)} 个非数值列: {non_numeric}")
else:
    print(f"\n✓ 所有列已转为数值类型，可直接输入模型!")

# 保存完整特征矩阵 (明细行级别)
df.to_csv("feature_matrix.csv", index=False, encoding="utf-8-sig")
print(f"\n✓ 已保存: feature_matrix.csv ({df.shape[0]}行 × {df.shape[1]}列)")

# =============================================================
# 第七步(可选): 按订单聚合 — 每订单一条记录
# =============================================================
# 有些模型更适合用"订单"作为基本单位来预测

print("\n" + "=" * 60)
print("第七步(可选): 订单级聚合")
print("=" * 60)

# 注意: 经过上面的处理，原始ID列已删除，但 Order ID 已在第四步被删除
# 所以我们需要从原始的 data_cleaned.csv 重新做一次订单级聚合
df_raw = pd.read_csv("data_cleaned.csv", encoding="gbk")
df_raw["Order Date"] = pd.to_datetime(df_raw["Order Date"], format="%d/%m/%Y")
df_raw["Ship Date"] = pd.to_datetime(df_raw["Ship Date"], format="%d/%m/%Y")

# 用订单ID分组，每订单汇总
# 先计算发货耗时，再聚合
df_raw["ShippingDays"] = (df_raw["Ship Date"] - df_raw["Order Date"]).dt.days

order_df = df_raw.groupby("Order ID", as_index=False).agg(
    OrderDate=("Order Date", "first"),
    ShipDate=("Ship Date", "first"),
    Segment=("Segment", "first"),
    Region=("Region", "first"),
    ShipMode=("Ship Mode", "first"),
    TotalSales=("Sales", "sum"),            # 订单总销售额 ← 预测目标
    ItemCount=("Sales", "count"),           # 订单包含多少件商品
    UniqueCategories=("Category", "nunique"),  # 涉及几类产品
    ShippingDays=("ShippingDays", "first"),     # 发货耗时(天)
)

# 从日期中提取时间特征
order_df["OrderYear"] = order_df["OrderDate"].dt.year
order_df["OrderMonth"] = order_df["OrderDate"].dt.month
order_df["OrderDay"] = order_df["OrderDate"].dt.day
order_df["OrderDayOfWeek"] = order_df["OrderDate"].dt.dayofweek
order_df["OrderQuarter"] = order_df["OrderDate"].dt.quarter
order_df.drop(columns=["OrderDate", "ShipDate"], inplace=True)

# 添加日期衍生特征
order_df["IsWeekend"] = (order_df["OrderDayOfWeek"] >= 5).astype(int)
order_df["IsHolidaySeason"] = order_df["OrderMonth"].isin([11, 12]).astype(int)
order_df["AvgItemPrice"] = order_df["TotalSales"] / order_df["ItemCount"]
order_df["LogTotalSales"] = np.log1p(order_df["TotalSales"])

# 对类别列做独热编码
for col in ["Segment", "Region", "ShipMode"]:
    dummies = pd.get_dummies(order_df[col], prefix=col, dtype=int)
    order_df = pd.concat([order_df, dummies], axis=1)
order_df.drop(columns=["Segment", "Region", "ShipMode", "Order ID"], inplace=True)

print(f"订单级特征矩阵: {order_df.shape[0]} 行 × {order_df.shape[1]} 列")
print("列名:", list(order_df.columns))

order_df.to_csv("feature_order.csv", index=False, encoding="utf-8-sig")
print("✓ 已保存: feature_order.csv (订单级)")

print("\n" + "=" * 60)
print("全部完成! 可直接用于模型训练的文件:")
print("=" * 60)
print("  1. feature_matrix.csv  — 明细行级别 (9800行)")
print("  2. feature_order.csv   — 订单级别   (约5000行)")
print("\n提示: 用 pandas 读取后即可训练模型:")
print('  import pandas as pd')
print('  df = pd.read_csv("feature_matrix.csv")')
print('  X = df.drop(columns=["Sales"])    # 特征')
print('  y = df["Sales"]                    # 目标变量')
