import pandas as pd

#usecols选取相应的列
data = pd.read_csv('2013_small.csv', usecols=[1, 2, 3, 4, 5, 7, 8, 9, 12])
states=['Pichincha','Cotopaxi','Chimborazo','Imbabura','Santo Domingo de los Tsachilas','Bolivar']
party=[]
#读入10个对应商店的数据为dataframe格式
for i in range(6):
    party.append(data.loc[data.state==states[i]])
    #下面一行代码可生成10个文件，对应相应的party
    party[i].to_csv('party%s.csv'%(str(i)))

print(party[0]['date'])


