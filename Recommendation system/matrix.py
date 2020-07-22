# -*- coding: utf-8 -*-
# @Author  : lfx
from numpy import *
from pylab import *
from numpy import *
import xlrd
import copy

class Martrix:

    def excel2m(path):  # 读excel数据转为矩阵函数
        '''

        :param path:  excel文件
        :return:      excel文件对应的矩阵
        '''
        data = xlrd.open_workbook(path)
        table = data.sheets()[0]  # 获取excel中第一个sheet表
        nrows = table.nrows  # 行数
        ncols = table.ncols  # 列数
        datamatrix = np.zeros((nrows - 1, ncols - 1))
        for x in range(1, ncols):
            cols = table.col_values(x)
            m = cols[1:]
            # print(m)
            cols1 = np.matrix(m)  # 把list转换为矩阵进行矩阵操作
            datamatrix[:, x - 1] = cols1  # 把数据进行存储
        return datamatrix

    def gradAscent(dataMat, K):
        '''

        :param dataMat:  待分解的矩阵
        :param K:  维度
        :return:  分解后的p、q
        '''
        dataMat = array(dataMat)
        print("original matrix:")
        print(dataMat)
        m, n = shape(dataMat)
        # print(m)
        # print(n)
        p = mat(random.random((m, K)))
        q = mat(random.random((K, n)))

        return dataMat, m, n, p, q

    def process(dataMat, K, m, n, p, q):
        alpha = 0.0002
        beta = 0.02
        maxCycles = 100000
        mutex = 1

        for i in range(m):  # 第i行
            for j in range(n):  # 第j列
                if dataMat[i, j] > 0:
                    error = dataMat[i, j]
                    for k in range(K):
                        error = error - p[i, k] * q[k, j]  # error=R[i,j]-R`[i,j]

                    for k in range(K):
                        p[i, k] = p[i, k] + alpha * (2 * error * q[k, j] - beta * p[i, k])
                        q[k, j] = q[k, j] + alpha * (2 * error * p[i, k] - beta * q[k, j])
        gradient = error
        loss = 0.0
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = 0.0
                    for k in range(K):
                        error = error + p[i, k] * q[k, j]
                    loss = (dataMat[i, j] - error) * (dataMat[i, j] - error)
                    for k in range(K):
                        loss = loss + beta * (p[i, k] * p[i, k] + q[k, j] * q[k, j]) / 2

        #print(loss)

        #gradient = Martrix.public_key(gradient)
        return gradient, loss, p, q
