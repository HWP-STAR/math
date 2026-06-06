import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('data.csv', encoding='gbk')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')

daily = df.set_index('Order Date').resample('D')['Sales'].sum()
daily = daily[daily.index >= '2022-01-01']
weekly = daily.resample('W-MON').sum()
monthly = daily.resample('ME').sum()
quarterly = daily.resample('QE').sum()
yearly = daily.resample('YE').sum()

print("=" * 60)
print("周期特征分析")
print("=" * 60)

# ============================================================
# 1. 自相关分析
# ============================================================
print("\n[1] 自相关函数 (ACF/PACF)")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(monthly.dropna(), lags=24, ax=axes[0, 0], title='Monthly Sales ACF (up to 24 lags)')
plot_pacf(monthly.dropna(), lags=24, ax=axes[0, 1], title='Monthly Sales PACF (up to 24 lags)')

plot_acf(weekly.dropna(), lags=52, ax=axes[1, 0], title='Weekly Sales ACF (up to 52 lags)')
plot_pacf(weekly.dropna(), lags=52, ax=axes[1, 1], title='Weekly Sales PACF (up to 52 lags)')

plt.tight_layout()
plt.savefig('cycle_acf_pacf.png', dpi=150)
plt.close()
print("  -> cycle_acf_pacf.png")

# ============================================================
# 2. 年度叠加对比 (月/周)
# ============================================================
print("\n[2] 年度叠加对比")

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['WeekOfYear'] = df['Order Date'].dt.isocalendar().week.astype(int)
df['Quarter'] = df['Order Date'].dt.quarter
df['DayOfWeek'] = df['Order Date'].dt.dayofweek

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 2a. 月度叠加 ---
monthly_by_year = df.groupby(['Year', 'Month'])['Sales'].sum().unstack(level=0)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, year in enumerate(monthly_by_year.columns):
    axes[0, 0].plot(monthly_by_year.index, monthly_by_year[year].values,
                    marker='o', color=colors[i], label=str(year), linewidth=1.5)
axes[0, 0].set_title('Monthly Sales: Year Over Year', fontsize=11)
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].legend(fontsize=9)
axes[0, 0].set_xticks(range(1, 13))
axes[0, 0].grid(alpha=0.3)

# --- 2b. 累计占比对比 ---
monthly_pct = monthly_by_year.div(monthly_by_year.sum(axis=0)) * 100
for i, year in enumerate(monthly_pct.columns):
    axes[0, 1].plot(monthly_pct.index, monthly_pct[year].values,
                    marker='s', color=colors[i], label=str(year), linewidth=1.5)
axes[0, 1].set_title('Monthly Sales Share: Year Over Year (%)', fontsize=11)
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Share of Annual Sales (%)')
axes[0, 1].legend(fontsize=9)
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].grid(alpha=0.3)

# --- 2c. 周度叠加 (平滑) ---
weekly_by_year = df.groupby(['Year', 'WeekOfYear'])['Sales'].sum().unstack(level=0)
for i, year in enumerate(weekly_by_year.columns):
    axes[1, 0].plot(weekly_by_year.index, weekly_by_year[year].values,
                    color=colors[i], label=str(year), linewidth=0.8, alpha=0.8)
axes[1, 0].set_title('Weekly Sales: Year Over Year', fontsize=11)
axes[1, 0].set_xlabel('Week of Year')
axes[1, 0].set_ylabel('Sales ($)')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# --- 2d. 季度叠加 ---
quarterly_by_year = df.groupby(['Year', 'Quarter'])['Sales'].sum().unstack(level=0)
x = np.arange(4) + 1
width = 0.2
for i, year in enumerate(quarterly_by_year.columns):
    axes[1, 1].bar(x + i * width, quarterly_by_year[year].values,
                   width, color=colors[i], label=str(year), alpha=0.8)
axes[1, 1].set_title('Quarterly Sales: Year Over Year', fontsize=11)
axes[1, 1].set_xlabel('Quarter')
axes[1, 1].set_ylabel('Sales ($)')
axes[1, 1].set_xticks(x + width * 1.5)
axes[1, 1].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cycle_year_overlay.png', dpi=150)
plt.close()
print("  -> cycle_year_overlay.png")

# ============================================================
# 3. 周内效应
# ============================================================
print("\n[3] 周内日分布")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# --- 3a. 按周几汇总 ---
dow_sales = df.groupby('DayOfWeek')['Sales'].agg(['sum', 'mean', 'count'])
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[0].bar(dow_labels, dow_sales['sum'], color='steelblue', edgecolor='white')
axes[0].set_title('Total Sales by Day of Week', fontsize=10)
axes[0].set_ylabel('Sales ($)')
for i, v in enumerate(dow_sales['sum']):
    axes[0].text(i, v, f'${v/1000:.1f}K', ha='center', va='bottom', fontsize=7)
axes[0].grid(alpha=0.3, axis='y')

# --- 3b. 年均按周几 ---
dow_yearly = df.groupby(['Year', 'DayOfWeek'])['Sales'].sum().unstack(level=0)
dow_yearly_normalized = dow_yearly.div(dow_yearly.sum(axis=0)) * 100
x = np.arange(7)
width = 0.2
for i, year in enumerate(dow_yearly_normalized.columns):
    axes[1].bar(x + i * width, dow_yearly_normalized[year].values,
                width, color=colors[i], label=str(year), alpha=0.8)
axes[1].set_title('Daily Share by Year (%)', fontsize=10)
axes[1].set_ylabel('Share of Weekly Sales (%)')
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(dow_labels)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, axis='y')

# --- 3c. 箱线图 ---
df.boxplot(column='Sales', by='DayOfWeek', ax=axes[2], patch_artist=True)
axes[2].set_title('Sales Distribution by Day of Week', fontsize=10)
axes[2].set_ylabel('Sales ($)')
axes[2].set_xticklabels(dow_labels)
axes[2].set_xlabel('')

plt.tight_layout()
plt.savefig('cycle_dayofweek.png', dpi=150)
plt.close()
print("  -> cycle_dayofweek.png")

# ============================================================
# 4. 月度/季度周期指数
# ============================================================
print("\n[4] 季节指数")

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 4a. 月度季节指数 ---
month_avg = df.groupby('Month')['Sales'].mean()
month_idx = month_avg / month_avg.mean() * 100
axes[0].fill_between(range(1, 13), month_idx.values, alpha=0.3, color='steelblue')
axes[0].plot(range(1, 13), month_idx.values, 'o-', color='steelblue', linewidth=2)
axes[0].axhline(100, color='red', ls='--', alpha=0.6, label='Baseline=100')
axes[0].set_title('Monthly Seasonal Index (Avg Sales Relative to Overall Mean)', fontsize=11)
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Seasonal Index (Mean=100)')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_names, rotation=30)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)
for i, v in enumerate(month_idx.values):
    axes[0].text(i + 1, v + 2, f'{v:.0f}', ha='center', va='bottom', fontsize=7)

# --- 4b. 月度销售额占比堆积 ---
month_pivot = df.groupby(['Year', 'Month'])['Sales'].sum().unstack(level=1)
month_pivot_pct = month_pivot.div(month_pivot.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(month_pivot_pct))
for m in range(1, 13):
    axes[1].bar(month_pivot_pct.index, month_pivot_pct[m].values,
                bottom=bottom, label=month_names[m - 1], width=0.6)
    bottom += month_pivot_pct[m].values
axes[1].set_title('Monthly Sales Share by Year (Stacked)', fontsize=11)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Share of Annual Sales (%)')
axes[1].legend(fontsize=6, ncol=3, loc='upper right')
axes[1].set_xticks(range(2022, 2026))

plt.tight_layout()
plt.savefig('cycle_seasonal_index.png', dpi=150)
plt.close()
print("  -> cycle_seasonal_index.png")

# ============================================================
# 5. 同比/环比增长率
# ============================================================
print("\n[5] 增长率分析")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# --- 5a. 月度环比 ---
monthly_ts = monthly.copy()
mom = monthly_ts.pct_change() * 100
axes[0, 0].plot(mom.index, mom.values, 'o-', color='steelblue', markersize=3, linewidth=0.8)
axes[0, 0].axhline(0, color='red', ls='--', alpha=0.5)
axes[0, 0].set_title('Month-over-Month Growth Rate (%)', fontsize=10)
axes[0, 0].set_ylabel('MoM (%)')
axes[0, 0].grid(alpha=0.3)

# --- 5b. 月度同比 ---
yoy = monthly_ts.pct_change(periods=12) * 100
axes[0, 1].plot(yoy.dropna().index, yoy.dropna().values, 'o-', color='coral', markersize=4, linewidth=1)
axes[0, 1].axhline(0, color='red', ls='--', alpha=0.5)
axes[0, 1].set_title('Year-over-Year Growth Rate (%)', fontsize=10)
axes[0, 1].set_ylabel('YoY (%)')
axes[0, 1].grid(alpha=0.3)

# --- 5c. 年度增长率 ---
yearly_ts = yearly.copy()
yoy_yearly = yearly_ts.pct_change() * 100
axes[1, 0].bar(yoy_yearly.index.year[1:], yoy_yearly.values[1:], color='mediumseagreen', width=0.4)
axes[1, 0].axhline(0, color='red', ls='--', alpha=0.5)
axes[1, 0].set_title('Year-over-Year Growth Rate (%)', fontsize=10)
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('YoY (%)')
for i, v in enumerate(yoy_yearly.values[1:]):
    axes[1, 0].text(yoy_yearly.index.year[1:][i], v + 1, f'{v:.1f}%',
                    ha='center', va='bottom', fontsize=9)
axes[1, 0].grid(alpha=0.3, axis='y')

# --- 5d. 累计销售额曲线 ---
cumulative = df.set_index('Order Date').sort_index()['Sales'].cumsum()
cumulative_by_year = {}
for year in range(2022, 2026):
    year_data = df[df['Year'] == year].set_index('Order Date').sort_index()
    cum = year_data['Sales'].cumsum()
    day_of_year = (cum.index - pd.Timestamp(f'{year}-01-01')).days
    cum_normalized = cum / cum.iloc[-1] * 100
    axes[1, 1].plot(day_of_year, cum_normalized.values, label=str(year), color=colors[year - 2022])
axes[1, 1].set_title('Normalized Cumulative Sales by Year (%)', fontsize=10)
axes[1, 1].set_xlabel('Day of Year')
axes[1, 1].set_ylabel('Cumulative Sales (%)')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cycle_growth.png', dpi=150)
plt.close()
print("  -> cycle_growth.png")

# ============================================================
# 6. 统计摘要
# ============================================================
print("\n" + "=" * 60)
print("周期特征统计摘要")
print("=" * 60)

print(f"\n--- 月度季节指数 ---")
for m in range(1, 13):
    val = month_idx[m]
    arrow = "↑" if val > 100 else "↓"
    print(f"  {month_names[m-1]:4s}: {val:.1f}  {arrow}")

print(f"\n--- 月度同比增速 (YoY) ---")
for date, val in yoy.dropna().items():
    print(f"  {date.strftime('%Y-%m')}: {val:+.2f}%")

print(f"\n--- 年度增速 ---")
for date, val in yoy_yearly.items():
    if pd.notna(val):
        print(f"  {date.year}: {val:+.2f}%")

print(f"\n--- 各年 Q4 占比 ---")
q4_share = {}
for year in range(2022, 2026):
    year_total = df[df['Year'] == year]['Sales'].sum()
    q4_total = df[(df['Year'] == year) & (df['Quarter'] == 4)]['Sales'].sum()
    q4_share[year] = (q4_total / year_total) * 100
    print(f"  {year}: Q4 = {q4_total:,.0f} / Total = {year_total:,.0f} = {q4_share[year]:.1f}%")

print(f"\n--- 周内分布 (总销售额占比) ---")
dow_pct = dow_sales['sum'] / dow_sales['sum'].sum() * 100
for i, label in enumerate(dow_labels):
    print(f"  {label}: {dow_pct.iloc[i]:.1f}%")

print("\n周期特征分析完成！")
