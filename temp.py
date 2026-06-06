import pandas as pd
df=pd.read_csv('feature_matrix.csv',encoding='gbk')
row=df.iloc[0]
print(row)

