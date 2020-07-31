# -*- coding: utf-8 -*-
# @Time    : 2020/07/12
# @Author  : lfx
from matplotlib import pyplot as plt
from Recommendation_system.server import Server
from Recommendation_system.matrix import Martrix
import numpy as np

class Driver :

    @staticmethod
    def readdata(data):
        '''
        :return: 构建矩阵
        '''
        t = Martrix()
        dat = t.excel2m(data)
        return dat

    @staticmethod
    def run(data1,data2,data3,K,step_s):
        '''
        :param data : 待分解的矩阵
        :param K : 维度
        :param step_s : 迭代次数
        :return: 最终矩阵
        '''
        M = Martrix()
        S = Server(M.get_paillier_public_key(),M.get_paillier_private_key())
        data1, m, n, data1_p, data1_q = M.gradAscent(data1, K)
        data2, m, n, data2_p, data2_q = M.gradAscent(data2, K)
        data3, m, n, data3_p, data3_q = M.gradAscent(data3, K)
        server_p, server_q = S.initial(K, m, n)
        data1_q = server_q
        loss_1 = loss_2 =loss_3 =  1
        step = 1
        loss_one = []
        loss_two = []
        loss_three = []
        while ((loss_1 > 0.01 or loss_2 > 0.01 or loss_3 > 0.01) and step < step_s):

            if loss_1 > 0.01:
                gradient_1, loss_1, p_1, q_1 = M.process(data1, K, m, n, data1_p, server_q)
                if step % 100 == 0:
                    loss_one.append(loss_1)
                server_q = S.updata(m, n, K, server_p, server_q, gradient_1)
            if loss_2 > 0.01:
                gradient_2, loss_2, p_2, q_2 = M.process(data2, K, m, n, data2_p, server_q)
                if step % 100 == 0:
                    loss_two.append(loss_2)
                server_q = S.updata(m, n, K, server_p, server_q, gradient_2)
            if loss_3 > 0.01:
                gradient_3, loss_3, p_3, q_3 = M.process(data3, K, m, n, data3_p, server_q)
                if step % 100 == 0:
                    loss_three.append(loss_3)
                server_q = S.updata(m, n, K, server_p, server_q, gradient_3)

            step += 1
        print("party0",p_1 * q_1)
        print("party1",p_2 * q_2)
        print("party3",p_3 * q_3)
        #print("loss0:",loss_one[-1])
        #print("loss1:", loss_two[-1])
        #print("loss2:", loss_three[-1])

        # print("loss1",loss_one)
        # print("loss2",loss_two)
        loss_one_log = np.log(np.array(loss_one))
        plt.plot(loss_one_log)
        plt.show()
        loss_two_log = np.log(np.array(loss_two))
        plt.plot(loss_two_log)
        plt.show()
        loss_three_log = np.log(np.array(loss_three))
        plt.plot(loss_three_log)
        plt.show()




