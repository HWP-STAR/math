import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data.csv', encoding='gbk')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')

# =============================================
# 1. 基本统计量
# =============================================
print("=" * 70)
print("1. 基本统计量与核心业务指标")
print("=" * 70)

sales = df['Sales']
print(f"\n销售额基本统计量:")
print(f"  均值 (Mean):     {sales.mean():.2f}")
print(f"  中位数 (Median): {sales.median():.2f}")
print(f"  标准差 (Std):    {sales.std():.2f}")
print(f"  最小值 (Min):    {sales.min():.2f}")
print(f"  最大值 (Max):    {sales.max():.2f}")

total_sales = df['Sales'].sum()
total_orders = df['Order ID'].nunique()
total_customers = df['Customer ID'].nunique()
total_products = df['Product ID'].nunique()
total_items = len(df)

print(f"\n核心业务指标:")
print(f"  总销售额 (Total Sales):       ${total_sales:,.2f}")
print(f"  总订单数 (Total Orders):      {total_orders}")
print(f"  有效客户数 (Unique Customers): {total_customers}")
print(f"  产品总数 (Unique Products):    {total_products}")
print(f"  订单明细总数 (Total Rows):    {total_items}")

# =============================================
# 2. 销售额时间趋势图
# =============================================
print("\n" + "=" * 70)
print("2. 销售额随时间变化趋势图（已保存为 png）")
print("=" * 70)

daily_sales = df.set_index('Order Date')['Sales'].resample('D').sum()
monthly_sales = df.set_index('Order Date')['Sales'].resample('ME').sum()
yearly_sales = df.set_index('Order Date')['Sales'].resample('YE').sum()

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(daily_sales.index, daily_sales.values, linewidth=0.5, color='steelblue')
axes[0].set_title('Daily Sales Trend')
axes[0].set_ylabel('Sales ($)')
axes[0].grid(alpha=0.3)

axes[1].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=1.5, color='coral')
axes[1].set_title('Monthly Sales Trend')
axes[1].set_ylabel('Sales ($)')
axes[1].grid(alpha=0.3)

axes[2].bar(yearly_sales.index.year, yearly_sales.values, color='mediumseagreen', width=0.5)
axes[2].set_title('Yearly Sales Trend')
axes[2].set_ylabel('Sales ($)')
axes[2].set_xlabel('Year')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('problem1_trend.png', dpi=150)
plt.close()
print("  -> problem1_trend.png (日/月/年趋势图)")

# =============================================
# 3. 多维度分布分析
# =============================================
print("\n" + "=" * 70)
print("3. 多维度分布分析（已保存为 png）")
print("=" * 70)

df['Year'] = df['Order Date'].dt.year
df['Quarter'] = df['Order Date'].dt.quarter
df['Month'] = df['Order Date'].dt.month

fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# --- 按年份 ---
year_sales = df.groupby('Year')['Sales'].sum()
axes[0, 0].bar(year_sales.index.astype(str), year_sales.values, color='steelblue', width=0.5)
axes[0, 0].set_title('Sales by Year')
axes[0, 0].set_ylabel('Total Sales ($)')
for i, v in enumerate(year_sales.values):
    axes[0, 0].text(i, v, f'${v/1000:.0f}K', ha='center', va='bottom', fontsize=8)
axes[0, 0].grid(alpha=0.3, axis='y')

# --- 按季度 ---
quarter_sales = df.groupby('Quarter')['Sales'].sum()
axes[0, 1].bar(quarter_sales.index.astype(str), quarter_sales.values, color='coral', width=0.5)
axes[0, 1].set_title('Sales by Quarter')
axes[0, 1].set_ylabel('Total Sales ($)')
for i, v in enumerate(quarter_sales.values):
    axes[0, 1].text(i, v, f'${v/1000:.0f}K', ha='center', va='bottom', fontsize=8)
axes[0, 1].grid(alpha=0.3, axis='y')

# --- 按月份 ---
month_sales = df.groupby('Month')['Sales'].sum()
axes[0, 2].bar(month_sales.index.astype(str), month_sales.values, color='mediumseagreen', width=0.5)
axes[0, 2].set_title('Sales by Month')
axes[0, 2].set_ylabel('Total Sales ($)')
for i, v in enumerate(month_sales.values):
    axes[0, 2].text(i, v, f'${v/1000:.0f}K', ha='center', va='bottom', fontsize=7)
axes[0, 2].grid(alpha=0.3, axis='y')

# --- 按区域 ---
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
colors_region = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
axes[1, 0].pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%',
               colors=colors_region, startangle=90)
axes[1, 0].set_title('Sales by Region')

# --- 按产品大类 ---
cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
colors_cat = ['#ff9999', '#66b3ff', '#99ff99']
axes[1, 1].pie(cat_sales.values, labels=cat_sales.index, autopct='%1.1f%%',
               colors=colors_cat, startangle=90)
axes[1, 1].set_title('Sales by Category')

# --- 按客户细分 ---
seg_sales = df.groupby('Segment')['Sales'].sum().sort_values(ascending=False)
colors_seg = ['#ff9999', '#66b3ff', '#99ff99']
axes[1, 2].pie(seg_sales.values, labels=seg_sales.index, autopct='%1.1f%%',
               colors=colors_seg, startangle=90)
axes[1, 2].set_title('Sales by Segment')

# --- 按发货模式 ---
ship_sales = df.groupby('Ship Mode')['Sales'].sum().sort_values(ascending=False)
axes[2, 0].barh(ship_sales.index, ship_sales.values, color='steelblue')
axes[2, 0].set_title('Sales by Ship Mode')
axes[2, 0].set_xlabel('Total Sales ($)')
for i, v in enumerate(ship_sales.values):
    axes[2, 0].text(v, i, f'${v/1000:.0f}K', ha='left', va='center', fontsize=8)
axes[2, 0].grid(alpha=0.3, axis='x')

# --- 销售额分布箱线图 (Category) ---
df.boxplot(column='Sales', by='Category', ax=axes[2, 1])
axes[2, 1].set_title('Sales Distribution by Category')
axes[2, 1].set_ylabel('Sales ($)')
axes[2, 1].grid(alpha=0.3, axis='y')

# --- 销售额分布箱线图 (Region) ---
df.boxplot(column='Sales', by='Region', ax=axes[2, 2])
axes[2, 2].set_title('Sales Distribution by Region')
axes[2, 2].set_ylabel('Sales ($)')
axes[2, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('problem1_distribution.png', dpi=150)
plt.close()
print("  -> problem1_distribution.png (多维度分布图)")

# =============================================
# 控制台输出汇总
# =============================================
print("\n" + "=" * 70)
print("各维度销售额汇总（总和）")
print("=" * 70)
print(f"\n按年份:\n{year_sales.to_string()}")
print(f"\n按季度:\n{quarter_sales.to_string()}")
print(f"\n按月份:\n{month_sales.to_string()}")
print(f"\n按区域:\n{region_sales.to_string()}")
print(f"\n按产品大类:\n{cat_sales.to_string()}")
print(f"\n按客户细分:\n{seg_sales.to_string()}")
print(f"\n按发货模式:\n{ship_sales.to_string()}")
print("\n问题1全部完成！")
