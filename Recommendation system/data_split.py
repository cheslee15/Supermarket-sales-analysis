# -*- coding: utf-8 -*-
# @Time    : 2020/07/25
# @Author  : lfx
path = r"../dataset/2013_big.csv"  # 改成自己的目录
import pandas as pd

data = pd.read_csv(path)
datas = data.iloc[:, 2:4]  # 提取目标咧第二列和第三咧
dex = set(datas["store_nbr"])  # store_nbr去重，作为结果的index
col = set(datas["item_nbr"])  # item_nbr 去重作为结果的columns
res = pd.DataFrame(index=dex, columns=col)  # 创建结果Dataframe
res = res.fillna(0)  # 用0填充所有位置
for i in datas.values:
    res.loc[i[0], i[1]] = 1  # 循环检索计数


def split(data, n):  # 分割文件的函数 n为份数，data为上一步
    step = len(data) // n  # 判断每一组的长度
    point = [i for i in range(len(res))][::step]  # 确认分割点

    if len(data) % n == 0:
        point.append(len(data))  # 如果刚好整分，就把最后一个位置加入位置
    else:
        point[-1] = (len(res))  # 如果有多余的，全部加入最后一组
    for i in range(n):  # 循环分割
        res.iloc[point[i]:point[i + 1], :].to_excel("big%s.xlsx" % i)  # 将结果保存为excel 用split+i明名
        print("******")


split(res, 3)  # 调用函数分割为三份量