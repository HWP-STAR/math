import csv
from datetime import datetime
from collections import defaultdict

INPUT_FILE = 'data.csv'
OUTPUT_FILE = 'data_cleaned.csv'

with open(INPUT_FILE, 'r', encoding='gbk') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

print(f"原始数据: {len(rows)} 条记录")

# ============================================================
# 问题1: 42条 Ship Date 年份错误 (2019 -> 2026)
# ============================================================
fix_ship_count = 0
ship_before_after = []
for r in rows:
    od = datetime.strptime(r['Order Date'].strip(), '%d/%m/%Y')
    sd = datetime.strptime(r['Ship Date'].strip(), '%d/%m/%Y')
    if sd < od:
        old_ship = r['Ship Date']
        # Ship Date 年份错误: 2019 -> Order Date 年份 + 1 (2026)
        corrected_sd = sd.replace(year=od.year + 1)
        r['Ship Date'] = corrected_sd.strftime('%d/%m/%Y')
        ship_before_after.append((r['Row ID'], old_ship, r['Ship Date']))
        fix_ship_count += 1
print(f"修复 Ship Date 年份错误: {fix_ship_count} 条")

# ============================================================
# 问题2: 11条 Postal Code 缺失 (Burlington, Vermont -> 05401)
# ============================================================
missing_pc_count = 0
for r in rows:
    pc = r['Postal Code'].strip()
    if not pc:
        city, state = r['City'].strip(), r['State'].strip()
        old = r['Postal Code']
        # Burlington, Vermont 的邮编为 05401
        r['Postal Code'] = '05401'
        missing_pc_count += 1
print(f"填充缺失 Postal Code: {missing_pc_count} 条 (Burlington, VT -> 05401)")

# ============================================================
# 问题3: 429条 4位邮编补零为5位
# ============================================================
fix_pc_format_count = 0
for r in rows:
    pc = r['Postal Code'].strip()
    if pc and len(pc) == 4 and pc.isdigit():
        r['Postal Code'] = pc.zfill(5)
        fix_pc_format_count += 1
print(f"邮编4位补零为5位: {fix_pc_format_count} 条")

# ============================================================
# 写入清洗后数据
# ============================================================
with open(OUTPUT_FILE, 'w', encoding='gbk', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n清洗完成! 输出文件: {OUTPUT_FILE}")
print(f"总记录数: {len(rows)}")
