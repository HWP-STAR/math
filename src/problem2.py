import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('data.csv', encoding='gbk')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')

df['Year'] = df['Order Date'].dt.year
df['Quarter'] = df['Order Date'].dt.quarter
df['Month'] = df['Order Date'].dt.month
df['DayOfWeek'] = df['Order Date'].dt.dayofweek
df['ShippingDays'] = (df['Ship Date'] - df['Order Date']).dt.days
df['IsHolidaySeason'] = df['Month'].isin([11, 12]).astype(int)

# order-level aggregation
order_cols = ['Order ID', 'Order Date', 'Year', 'Quarter', 'Month', 'DayOfWeek',
              'Ship Date', 'ShippingDays', 'IsHolidaySeason',
              'Customer ID', 'Segment', 'Country', 'Region']
order_df = df.groupby('Order ID', as_index=False).agg(
    OrderDate=('Order Date', 'first'),
    Year=('Year', 'first'),
    Quarter=('Quarter', 'first'),
    Month=('Month', 'first'),
    DayOfWeek=('DayOfWeek', 'first'),
    ShippingDays=('ShippingDays', 'first'),
    IsHolidaySeason=('IsHolidaySeason', 'first'),
    Segment=('Segment', 'first'),
    Region=('Region', 'first'),
    ShipMode=('Ship Mode', 'first'),
    OrderSales=('Sales', 'sum'),
    ItemCount=('Sales', 'count'),
    UniqueCategories=('Category', 'nunique'),
)
order_df['AvgItemPrice'] = order_df['OrderSales'] / order_df['ItemCount']

# customer-level
customer_df = df.groupby('Customer ID', as_index=False).agg(
    Segment=('Segment', 'first'),
    Region=('Region', 'first'),
    TotalSpent=('Sales', 'sum'),
    OrderCount=('Order ID', 'nunique'),
    ItemCount=('Sales', 'count'),
)

def eta_squared(groups):
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((x - grand_mean) ** 2 for x in all_data)
    return ss_between / ss_total if ss_total != 0 else 0

def run_anova(data, group_col, value_col):
    groups = [g[value_col].values for _, g in data.groupby(group_col, observed=True) if len(g) > 1]
    if len(groups) < 2:
        return None
    f, p = stats.f_oneway(*groups)
    return {'Variable': group_col, 'F': f, 'p': p, 'eta2': eta_squared(groups),
            'sig': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}

# ============================================================
# 1. 多层次方差分析
# ============================================================
print("=" * 70)
print("1. 多层次方差分析 (ANOVA)")
print("=" * 70)

print("\n--- A. 明细行级别: 单行销售额差异分析 ---")
print("    (说明各因素对单笔交易金额的影响)")
item_vars = ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode',
             'Year', 'Quarter', 'Month', 'IsHolidaySeason']
item_results = [r for r in [run_anova(df, c, 'Sales') for c in item_vars] if r is not None]
item_results.sort(key=lambda x: x['eta2'], reverse=True)
for r in item_results:
    print(f"  {r['Variable']:20s}  F={r['F']:8.2f}  p={r['p']:.6f}  η²={r['eta2']:.4f}  {r['sig']}")

print("\n--- B. 订单级别: 订单总销售额差异分析 ---")
order_vars = ['ShipMode', 'Segment', 'Region', 'Year', 'Quarter', 'Month', 'IsHolidaySeason']
order_results = [r for r in [run_anova(order_df, c, 'OrderSales') for c in order_vars] if r is not None]
order_results.sort(key=lambda x: x['eta2'], reverse=True)
for r in order_results:
    print(f"  {r['Variable']:20s}  F={r['F']:8.2f}  p={r['p']:.6f}  η²={r['eta2']:.4f}  {r['sig']}")

print("\n--- C. 客户级别: 客户总消费差异分析 ---")
cust_vars = ['Segment', 'Region']
cust_results = [r for r in [run_anova(customer_df, c, 'TotalSpent') for c in cust_vars] if r is not None]
cust_results.sort(key=lambda x: x['eta2'], reverse=True)
for r in cust_results:
    print(f"  {r['Variable']:20s}  F={r['F']:8.2f}  p={r['p']:.6f}  η²={r['eta2']:.4f}  {r['sig']}")

# ============================================================
# 2. 数值变量相关性 (订单级)
# ============================================================
print("\n" + "=" * 70)
print("2. 订单级数值变量 Pearson 相关性")
print("=" * 70)
order_num = order_df[['OrderSales', 'ItemCount', 'AvgItemPrice',
                       'ShippingDays', 'Month', 'Quarter', 'Year']].dropna()
corr = order_num.corr(method='pearson')['OrderSales'].drop('OrderSales').sort_values(ascending=False)
for var, val in corr.items():
    _, p = stats.pearsonr(order_num[var], order_num['OrderSales'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {var:15s}  r={val:.4f}  p={p:.6f}  {sig}")

# ============================================================
# 3. 可视化
# ============================================================
print("\n" + "=" * 70)
print("3. 生成可视化图表")
print("=" * 70)

fig = plt.figure(figsize=(20, 18))

# --- 3a. 订单级相关系数热力图 ---
ax1 = fig.add_subplot(4, 4, 1)
corr_matrix = order_num.corr(method='pearson')
im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_xticks(range(len(corr_matrix.columns)))
ax1.set_yticks(range(len(corr_matrix.columns)))
ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=7)
ax1.set_yticklabels(corr_matrix.columns, fontsize=7)
ax1.set_title('Order-Level Correlation Heatmap', fontsize=9)
plt.colorbar(im, ax=ax1, shrink=0.8)

# --- 3b. 品类 - 销售额箱线图 ---
ax2 = fig.add_subplot(4, 4, 2)
df.boxplot(column='Sales', by='Category', ax=ax2, patch_artist=True,
           boxprops=dict(alpha=0.7))
ax2.set_title('Sales Distribution by Category', fontsize=9)
ax2.set_ylabel('Sales ($)')
ax2.set_xlabel('')

# --- 3c. 子品类 - 销售额对比 ---
ax3 = fig.add_subplot(4, 4, 3)
subcat_avg = df.groupby('Sub-Category')['Sales'].mean().sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(subcat_avg)))
ax3.barh(range(len(subcat_avg)), subcat_avg.values, color=colors)
ax3.set_yticks(range(len(subcat_avg)))
ax3.set_yticklabels(subcat_avg.index, fontsize=6)
ax3.set_title('Avg Sales by Sub-Category', fontsize=9)
ax3.set_xlabel('Avg Sales ($)')

# --- 3d. 子品类 - ANOVA eta² 条形图 ---
ax4 = fig.add_subplot(4, 4, 4)
eta_vals = [r['eta2'] for r in item_results]
var_names = [r['Variable'] for r in item_results]
colors_eta = plt.cm.Reds(np.linspace(0.3, 0.9, len(eta_vals)))
bars = ax4.barh(range(len(eta_vals)), eta_vals, color=colors_eta)
ax4.set_yticks(range(len(eta_vals)))
ax4.set_yticklabels(var_names, fontsize=8)
ax4.set_title("Effect Size (η²) on Item Sales", fontsize=9)
ax4.set_xlabel('η²')
for bar, v in zip(bars, eta_vals):
    ax4.text(v + 0.002, bar.get_y() + bar.get_height()/2, f'{v:.4f}',
             ha='left', va='center', fontsize=7)

# --- 3e. 区域 - 订单总销售额 ---
ax5 = fig.add_subplot(4, 4, 5)
region_stats = order_df.groupby('Region')['OrderSales'].agg(['sum', 'mean', 'count'])
x = range(len(region_stats))
ax5.bar(x, region_stats['sum'], color='steelblue', alpha=0.8, label='Total Sales')
ax5_twin = ax5.twinx()
ax5_twin.plot(x, region_stats['mean'], 'ro-', markersize=6, label='Avg Order Sales')
ax5.set_xticks(x)
ax5.set_xticklabels(region_stats.index, fontsize=8)
ax5.set_title('Sales by Region', fontsize=9)
ax5.set_ylabel('Total Sales ($)')
ax5_twin.set_ylabel('Avg Order Sales ($)')
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

# --- 3f. 客户细分 ---
ax6 = fig.add_subplot(4, 4, 6)
seg_stats = order_df.groupby('Segment')['OrderSales'].agg(['sum', 'mean', 'count'])
x = range(len(seg_stats))
ax6.bar(x, seg_stats['sum'], color='coral', alpha=0.8, label='Total Sales')
ax6_twin = ax6.twinx()
ax6_twin.plot(x, seg_stats['mean'], 'bo-', markersize=6, label='Avg Order Sales')
ax6.set_xticks(x)
ax6.set_xticklabels(seg_stats.index, fontsize=8)
ax6.set_title('Sales by Segment', fontsize=9)
ax6.set_ylabel('Total Sales ($)')
ax6_twin.set_ylabel('Avg Order Sales ($)')
lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

# --- 3g. 发货模式 ---
ax7 = fig.add_subplot(4, 4, 7)
ship_stats = order_df.groupby('ShipMode')['OrderSales'].agg(['sum', 'mean', 'count'])
x = range(len(ship_stats))
ax7.bar(x, ship_stats['sum'], color='mediumseagreen', alpha=0.8, label='Total Sales')
ax7_twin = ax7.twinx()
ax7_twin.plot(x, ship_stats['mean'], 'ro-', markersize=6, label='Avg Order Sales')
ax7.set_xticks(x)
ax7.set_xticklabels(ship_stats.index, fontsize=7)
ax7.set_title('Sales by Ship Mode', fontsize=9)
ax7.set_ylabel('Total Sales ($)')
ax7_twin.set_ylabel('Avg Order Sales ($)')
lines1, labels1 = ax7.get_legend_handles_labels()
lines2, labels2 = ax7_twin.get_legend_handles_labels()
ax7.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

# --- 3h. 产品大类子品类占比 ---
ax8 = fig.add_subplot(4, 4, 8)
cat_subcat = df.groupby(['Category', 'Sub-Category'])['Sales'].sum().unstack(level=0)
cat_subcat.plot(kind='barh', stacked=True, ax=ax8, colormap='Set2')
ax8.set_title('Sub-Category Sales Composition', fontsize=9)
ax8.set_xlabel('Total Sales ($)')
ax8.legend(fontsize=6, loc='lower right')

# --- 3i. 月度销售额趋势 ---
ax9 = fig.add_subplot(4, 4, 9)
monthly = df.set_index('Order Date').resample('ME')['Sales'].sum()
ax9.plot(monthly.index, monthly.values, '-o', markersize=3, color='steelblue', linewidth=1)
ax9.axvline(pd.Timestamp('2023-01-01'), color='gray', ls='--', alpha=0.5)
ax9.axvline(pd.Timestamp('2024-01-01'), color='gray', ls='--', alpha=0.5)
ax9.axvline(pd.Timestamp('2025-01-01'), color='gray', ls='--', alpha=0.5)
ax9.set_title('Monthly Sales Trend', fontsize=9)
ax9.set_ylabel('Sales ($)')
ax9.grid(alpha=0.3)

# --- 3j. 季度销售额 ---
ax10 = fig.add_subplot(4, 4, 10)
qtr_year = df.groupby(['Year', 'Quarter'])['Sales'].sum().unstack(level=1)
qtr_year.plot(kind='bar', ax=ax10, colormap='Set2', edgecolor='white')
ax10.set_title('Quarterly Sales by Year', fontsize=9)
ax10.set_ylabel('Sales ($)')
ax10.set_xlabel('Year')
ax10.legend(fontsize=7)
ax10.tick_params(axis='x', rotation=0)

# --- 3k. 订单商品数与销售额散点 ---
ax11 = fig.add_subplot(4, 4, 11)
ax11.scatter(order_df['ItemCount'], order_df['OrderSales'],
             alpha=0.3, s=3, c='steelblue')
z = np.polyfit(order_df['ItemCount'], order_df['OrderSales'], 1)
p = np.poly1d(z)
x_sorted = np.sort(order_df['ItemCount'])
ax11.plot(x_sorted, p(x_sorted), 'r--', linewidth=1)
ax11.set_title('Item Count vs Order Sales', fontsize=9)
ax11.set_xlabel('Items per Order')
ax11.set_ylabel('Order Sales ($)')
ax11.grid(alpha=0.3)

# --- 3l. 发货天数 vs 订单销售额 ---
ax12 = fig.add_subplot(4, 4, 12)
ship_binned = order_df.groupby('ShippingDays')['OrderSales'].mean().reset_index()
ax12.bar(ship_binned['ShippingDays'], ship_binned['OrderSales'],
         width=0.8, color='coral', alpha=0.7)
ax12.set_title('Avg Order Sales by Shipping Days', fontsize=9)
ax12.set_xlabel('Shipping Days')
ax12.set_ylabel('Avg Order Sales ($)')
ax12.grid(alpha=0.3, axis='y')

# --- 3m. 客户消费分布 ---
ax13 = fig.add_subplot(4, 4, 13)
customer_df['TotalSpent'].plot(kind='hist', bins=50, ax=ax13, color='steelblue', edgecolor='white')
ax13.axvline(customer_df['TotalSpent'].mean(), color='red', ls='--', label=f'Mean={customer_df["TotalSpent"].mean():.0f}')
ax13.set_title('Customer Total Spend Distribution', fontsize=9)
ax13.set_xlabel('Total Spend ($)')
ax13.set_ylabel('Customer Count')
ax13.legend(fontsize=7)

# --- 3n. 客户消费 vs 订单频次 ---
ax14 = fig.add_subplot(4, 4, 14)
ax14.scatter(customer_df['OrderCount'], customer_df['TotalSpent'],
             alpha=0.3, s=3, c='mediumseagreen')
ax14.set_title('Order Count vs Total Spend (per Customer)', fontsize=9)
ax14.set_xlabel('Order Count')
ax14.set_ylabel('Total Spend ($)')
ax14.grid(alpha=0.3)

# --- 3o. 区域销售额地图(模拟) ---
ax15 = fig.add_subplot(4, 4, 15)
region_map = {
    'West': 710220, 'East': 669519,
    'Central': 492647, 'South': 389151
}
regions = list(region_map.keys())
values = list(region_map.values())
colors_map = plt.cm.YlOrRd(np.array(values) / max(values))
bars = ax15.bar(regions, values, color=colors_map, edgecolor='white')
ax15.set_title('Total Sales by Region', fontsize=9)
ax15.set_ylabel('Sales ($)')
for bar, v in zip(bars, values):
    ax15.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
              f'${v/1000:.0f}K', ha='center', va='bottom', fontsize=8)

# --- 3p. 年末旺季 vs 其他 ---
ax16 = fig.add_subplot(4, 4, 16)
holiday_stats = order_df.groupby('IsHolidaySeason')['OrderSales'].agg(['sum', 'mean'])
labels = ['Other Months', 'Nov-Dec (Holiday)']
ax16.bar(labels, holiday_stats['sum'], color=['#66b3ff', '#ff9999'], edgecolor='white')
ax16.set_title('Holiday Season vs Other Months', fontsize=9)
ax16.set_ylabel('Total Sales ($)')
for i, v in enumerate(holiday_stats['sum']):
    ax16.text(i, v, f'${v/1000:.0f}K\n(avg ${holiday_stats["mean"].iloc[i]:.0f})',
              ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('problem2_correlation.png', dpi=150)
plt.close()
print("  -> problem2_correlation.png")

# ============================================================
# 4. 按维度汇总统计
# ============================================================
print("\n" + "=" * 70)
print("4. 订单级维度统计")
print("=" * 70)
for col in order_vars:
    grouped = order_df.groupby(col)['OrderSales'].agg(['count', 'mean', 'std', 'sum', 'median'])
    print(f"\n{col}:")
    print(grouped.to_string())

print("\n问题2全部完成！")
