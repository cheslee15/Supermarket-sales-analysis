# -*- coding: utf-8 -*-
# @Author  : lfx

from numpy import *
from pylab import *
from numpy import *
import xlrd
from phe import paillier

class Server:
    def __init__(self):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=1024)  # 同态加密公钥，秘钥
    def get_paillier_public_key(self):
        return self.public_key

    def get_paillier_private_key(self):
        return self.private_key

    def initial(K, m, n):
        '''
        :param K:  维度
        :param m:  原始矩阵的行
        :param n:  原始矩阵的列
        :return:   服务器端初始化后的p、q矩阵
        '''
        p = mat(random.random((m, K)))
        q = mat(random.random((K, n)))
        return p, q

    def updata(m, n, K, p, q, error):
        '''

        :param m:  原始矩阵的行
        :param n:  原始矩阵的列
        :param K:  维度
        :param p:  更新前的p
        :param q:  更新前的q
        :param error:  梯度
        :return:  更新后提供下载的q矩阵
        '''
        alpha = 0.0002
        beta = 0.02
        for i in range(m):  # 第i行
            for j in range(n):  # 第j列
                for k in range(K):
                    p[i, k] = p[i, k] + alpha * (2 * error * q[k, j] - beta * p[i, k])
                    q[k, j] = q[k, j] + alpha * (2 * error * p[i, k] - beta * q[k, j])
        return q



