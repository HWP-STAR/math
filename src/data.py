import pandas as pd

df = pd.read_csv('data.csv', encoding='gbk')

print("=" * 60)
print("数据概览")
print("=" * 60)
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
print()

print("=" * 60)
print("列名与数据类型")
print("=" * 60)
print(df.dtypes)
print()

print("=" * 60)
print("前5行")
print("=" * 60)
print(df.head())
print()

print("=" * 60)
print("后5行")
print("=" * 60)
print(df.tail())
print()

print("=" * 60)
print("缺失值统计")
print("=" * 60)
print(df.isnull().sum())
print()

print("=" * 60)
print("描述性统计 (数值列)")
print("=" * 60)
print(df.describe())
print()

print("=" * 60)
print("描述性统计 (类别列)")
print("=" * 60)
print(df.describe(include='object'))
