import csv,datetime
import pandas as pd
import numpy as np

df_train1 = pd.read_csv(
    'train.csv', usecols=[1, 2, 3, 4],
    nrows=10000000,
)
df1 = df_train1.loc[df_train1.store_nbr==6]
#df1 = df1.loc[df_train1.store_nbr>=5]
df1 = df1.loc[df1.item_nbr<=122418]
df1 = df1.loc[df1.item_nbr>=119193]
del df_train1
df_train2 = pd.read_csv(
    'train.csv', usecols=[1, 2, 3, 4],
    nrows=11000000,skiprows=range(1, 10000000)
)
df2 = df_train2.loc[df_train2.store_nbr==6]
#df2 = df2.loc[df_train2.store_nbr>=5]
df2 = df2.loc[df2.item_nbr<=122418]
df2 = df2.loc[df2.item_nbr>=119193]
del df_train2
df_items = pd.read_csv(
    'items.csv'
)
df_stores = pd.read_csv(
    'stores.csv'
)
df2=pd.concat([df1,df2],ignore_index=True)
df_part1 = df2.loc[df2.date=='2013-01-01']
for i in range(1,13):
    for j in [1,2,4,5,7,8]:
        if i==1 and j==1:
            continue
        tmp = df2.loc[df2.date=='2013-%s-0%s'%(str(i).zfill(2),str(j))]
        df_part1=pd.concat([df_part1,tmp],ignore_index=True)
df_part1=pd.merge(df_part1,df_items)
df_part1=pd.merge(df_part1,df_stores).sort_values('date')

df_part2 = df2.loc[df2.date=='2013-01-03']
for i in range(1,13):
    for j in [3,6,9]:
        if i==1 and j==3:
            continue
        tmp = df2.loc[df2.date=='2013-%s-0%s'%(str(i).zfill(2),str(j))]
        df_part2=pd.concat([df_part2,tmp],ignore_index=True)
df_part2=pd.merge(df_part2,df_items)
df_part2=pd.merge(df_part2,df_stores).sort_values('date')

df_part3=pd.concat([df_part1,df_part2])
mean_sales=df_part3.groupby(['store_nbr','item_nbr'])['unit_sales'].mean()
def is_hot(series):
    global mean_sales
    sales=series['unit_sales']
    if sales > mean_sales[(series['store_nbr'],series['item_nbr'])]*1.2:
        return 1
    else:
        return 0

df_part3['is_hot']=df_part3.apply(is_hot,axis=1)
print(df_part3)
print(list(mean_sales.items()))
#print(mean_sales[(6, 121964)])
df_part3.to_csv('2013_party2.csv')