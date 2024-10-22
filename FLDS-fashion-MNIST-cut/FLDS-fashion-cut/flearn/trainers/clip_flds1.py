import bisect

import numpy as np
from tqdm import tqdm, trange
import math
from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import clip, sparsify
from flearn.utils.priv_utils import sampling_randomizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from flearn.utils.priv_utils import one_laplace
from flearn.utils.utils import expMechanism

class Server(BaseFedarated):
    '''
    SS-FL-V2
    '''
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train (SS-Double)')
        #通过数据集、模参进行训练，获得梯度
        self.inner_opt = GradientDescentOptimizer(learning_rate=params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.clip_C = self.norm
        self.m_p = self.clients_per_round / self.mp_rate #相当于论文中的n_p
        print("Setting the padding size for each dimension with ", self.m_p)
        self.em_s = self.clients_per_round /self.rate
        self.sample = int( (self.dim_model + self.dim_y)/self.rate) #elf.rate表示每一行抽取多少个维度，这里为7850/50
        print("Randomly sampling {} dimensions".format(self.sample))
        self.choice_list = []

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened,iter,select_list,clip_vector,clip,eps):

        #print("查看控制向量：{}".format(len(select_list)))
        """
        flattened:用户梯度向量    iter：epoch数目     select_list：控制向量    clip_vector:剪切向量    eps：隐私预算
        本地加噪处理：分两种情况（1.第一次epoch,如何确定剪切值。2.剩余的epoch,如何确定剪切值。
             1.情况一：先计算分簇的数目clusters
                      将给定的剪切值clip均匀的划分为clusters段
                      得到控制向量中为1的索引列表select_index
                      执行len(flattened)次循环：
                          如果：索引在select_index中：
                             确定剪切值，根据剪切值添加噪音
                          如果：索引不在select_index中：
                             对应的梯度值设为0
             2.情况二：
             得到控制向量中为1的索引列表select_index
                      执行len(flattened)次循环：
                          如果：索引在select_index中：
                             根据clip_vector向量得到剪切值，根据剪切值添加噪音
                          如果：索引不在select_index中：
                             对应的梯度值设为0
        """
        """
        if iter == 0:
            select_index = np.where(np.array(select_list) == 1)
            for i in range(len(flattened)):
                if i in select_index[0]:
                    flattened[i] = np.clip(flattened[i], -clip, clip) + one_laplace(eps, 2 * clip)
                else:
                    flattened[i] = 0
                if iter == 0:
            clusters = math.ceil(math.log(7850, math.pow(2, 1)))  # 确定簇数
            clip_list = [clip * x / clusters for x in range(1, clusters + 1, 1)]  # 根据簇数将剪切值进行切分
            ##astype 出错，astype 是numpy中的对象
            # print("查看第一次epoch的剪切值的列表：{}".format(clip_list))
            select_index = np.where(np.array(select_list) == 1)  # 返回控制向量中为以的索引
            for i in range(len(flattened)):
                if i in select_index[0]:
                    clip_list_score = [np.exp(-abs(abs(flattened[i]) - x) + 2 * min(abs(flattened[i]) - x, 0)) for x in clip_list]  # 给每个剪切值打分
                    select_clip_index = expMechanism(clip_list_score, 0.5 * eps, 1)  # 按指数机制返回剪切值的索引
                    print("查看选中的区间：{}".format(select_clip_index))
                    flattened[i] = np.clip(flattened[i], -clip_list[select_clip_index], clip_list[select_clip_index]) + one_laplace(0.5 * eps, 2 * clip_list[select_clip_index])
                else:
                    flattened[i] = 0
        """
        if iter == 0:
            # clusters = math.ceil(math.log(7850, math.pow(2, 1)))  # 确定簇数
            # clip_list = [clip * x / clusters for x in range(1, clusters + 1, 1)]  # 根据簇数将剪切值进行切分
            ##astype 出错，astype 是numpy中的对象
            # print("查看第一次epoch的剪切值的列表：{}".format(clip_list))
            select_index = np.where(np.array(select_list) == 1)  # 返回控制向量中为以的索引
            for i in range(len(flattened)):
                if i in select_index[0]:
                    #clip_list_score = [np.exp(-abs(abs(flattened[i]) - x) + 2 * min(abs(flattened[i]) - x, 0)) for x in clip_list]  # 给每个剪切值打分
                    #select_clip_index = expMechanism(clip_list_score, 0.5 * eps, 1)  # 按指数机制返回剪切值的索引
                    #print("查看选中的区间：{}".format(select_clip_index))
                    flattened[i] = np.clip(flattened[i], -clip, clip) + one_laplace(eps, 2 * clip)
                else:
                    flattened[i] = 0
        else:
            select_index = np.where(np.array(select_list) == 1)
            for i in range(len(flattened)):
                if i in select_index[0]:
                    flattened[i]=np.clip(flattened[i],-clip_vector[i],clip_vector[i])+one_laplace(eps,2*clip_vector[i])
                else:
                    flattened[i] = 0
        return flattened

    def server_process(self, messages):
        '''
        basic aggregate, but enlarge the learning rate when Top-k is applied
        '''
        return self.aggregate_p(messages)
