# -*- coding: utf-8 -*-
# @Time    : 2020/07/12
# @Author  : lfx
from matplotlib import pyplot as plt
from Recommendation_system.server import Server
from Recommendation_system.matrix import Martrix
import numpy as np

class Driver :
    def readdata(data):
        dat = Martrix.excel2m(data)
        return dat

    def run(data1,data2,K,step_s):
        data1, m, n, data1_p, data1_q = Martrix.gradAscent(data1, K)
        data2, m, n, data2_p, data2_q = Martrix.gradAscent(data2, K)
        server_p, server_q = Server.initial(K, m, n)
        data1_q = server_q
        loss_1 = loss_2 = 1
        step = 1
        loss_one = []
        loss_two = []
        while ((loss_1 > 0.01 or loss_2 > 0.01) and step < step_s):

            if loss_1 > 0.01:
                gradient_1, loss_1, p_1, q_1 = Martrix.process(data1, K, m, n, data1_p, server_q)
                if step % 100 == 0:
                    loss_one.append(loss_1)
                server_q = Server.updata(m, n, K, server_p, server_q, gradient_1)
            if loss_2 > 0.01:
                gradient_2, loss_2, p_2, q_2 = Martrix.process(data2, K, m, n, data2_p, server_q)
                if step % 100 == 0:
                    loss_two.append(loss_2)
                server_q = Server.updata(m, n, K, server_p, server_q, gradient_2)
            step += 1

        # print("loss1",loss_one)
        # print("loss2",loss_two)
        loss_one_log = np.log(np.array(loss_one))
        plt.plot(loss_one_log)
        plt.show()
        loss_two_log = np.log(np.array(loss_two))
        plt.plot(loss_two_log)
        plt.show()
        '''
        print("p1:", p_1)
        print("q1", q_1)
        print("t1", p_1 * q_1)
        print("p2:", p_2)
        print("q2", q_2)
        print("t2", p_2 * q_2)
        '''




