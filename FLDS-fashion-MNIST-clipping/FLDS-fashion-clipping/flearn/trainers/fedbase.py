import bisect

import numpy as np
import math
import tensorflow as tf
from tqdm import trange, tqdm
import random

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics

from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import transform
from flearn.utils.utils import generater_random_matrix
from flearn.utils.priv_utils import new_randomizer
from flearn.utils.utils import every_cluster_clip
from flearn.utils.utils import image

class BaseFedarated(object):
    def __init__(self, params, learner, data):
        for key, val in params.items():
            setattr(self, key, val)

        # create worker nodes
        tf.reset_default_graph() #清除默认图形堆栈，暂略。属于tensorflow的值
        self.client_model = learner(
            *params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(data, self.dataset, self.model,
                                          self.client_model)  #6000个客户端初始化对象
        self.latest_model = self.client_model.get_params() #这个是模型参数
        #print("查看初始的模型参数值范围：{}".format(self.latest_model))
        self.dim_model, self.dim_x, self.dim_y = self.setup_dim(
            self.dataset, self.model)

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        # self.client_model.close()
        pass

    ##################################SET UP####################################
    def setup_dim(self, dataset_name, model_name):
        if model_name == 'mclr':
            if dataset_name == 'adult':
                return 104*2, 104, 2
            elif dataset_name == 'mnist': #这里为何设置成这个值呢？
                return 784*10, 784, 10
        else:
            raise ValueError("Unknown dataset and model")

    def setup_clients(self, dataset, dataset_name, model_name, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''

        users, groups, train_data, test_data = dataset #在npsgd当中，groups的值是空的。这里的users实际就是一系列ID，train_data和test_data就是实际的数据
        #print("这里用来测试groups的值：{}".format(groups))
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(id=u, group=g, dataset_name=dataset_name, model_name=model_name,  # noqa: E501
                              train_data=train_data[u], eval_data=test_data[u], model=model) for u, g in zip(users, groups)]  # noqa: E501
        return all_clients


    #################################TRAINING#################################
    def train_grouping(self):
        #------------------------------------------------------------------------------------------------------------
        count_iter = 0
        clip_vector =[0]*7850
        for global_epoch in range(self.num_rounds): #全体的数据集epoch次数
            # loop through mini-batches of clients
            #for iter in range(0, len(self.clients), self.clients_per_round): #default=1000
            for iter in range(0, len(self.clients), 1000):
                if count_iter % self.eval_every == 0:
                    self.evaluate(count_iter)
                random.shuffle(self.clients)
	selected_clients = self.clients[iter: iter + self.clients_per_round]
                #新选择的用户数目，这一行可以自己调节
                #new_selected_clients = selected_clients[:942]
                csolns = []
                for client_id, c in enumerate(selected_clients):
                    # distribute global model
                    c.set_params(self.latest_model)
                    # local iteration on full local batch of client c
                    num_samples, grads = c.get_grads(7850)
                    # track computational cost
                    #self.metrics.update(rnd=i, cid=c.id, stats=stats)
                    # local update
                    #model_updates = [u - v for (u, v) in zip(soln[1], self.latest_model)]
                    # aggregate local update
                    csolns.append(grads)

                ########################## sever process #########################
                #步骤一：生成随机矩阵和噪音矩阵
                random_index_matrix= generater_random_matrix(942, 7850, 6, 50)
                print("查看用户选择矩阵的行的长度：{}".format(len(random_index_matrix[0])))
                print("查看一行有多少个1：{}".format(np.sum(random_index_matrix[0]==1)))
                ########################## local process #########################
                #步骤二：根据随机矩阵从temper当中抽取梯度,存放到choices_gradients当中
                temper = []                                   #temper当中最终存放的是1000个clients的梯度，
                print("查看epoch数：{}".format(global_epoch))
                for id,csoln in enumerate(csolns):
                    flattened = process_grad(csoln)         #将用户的梯度展平为一维数组
                    processed_update=self.local_process(flattened,count_iter,random_index_matrix[id],clip_vector,self.clip_C,self.epsilon)
                    #(self, flattened, iter, select_list, clip_vector, clip, eps)
                    temper.append(processed_update)
                ########################## sever process #########################
                #步骤四：每个维度进行求和，平均
                mean_sum_noise_gradients_shape = np.sum(temper,axis=0) / 6
                # b = list(np.argsort(np.abs(mean_sum_noise_gradients_shape)))
                # count_output = 0
                # for j in b:  #
                #     print(j, end=', ')
                #     count_output += 1  # 开始计数
                #     if count_output % 30 == 0:  # 每10个换行
                #         print(end='\n')

                # if count_iter in range(0,7):
                #     image(mean_sum_noise_gradients_shape)
                #聚类选择剪切值
                """
                服务器端在每次epoch结束的那次迭代中确定剪切向量 
                """
                if  ( count_iter == 0 or iter == (len(self.clients)-self.clients_per_round)):
                    if count_iter==0:
                        num_clusters = math.ceil(math.log(7850, math.pow(2, 1)))  ##计算簇数
                    else:
                        num_clusters = math.ceil(math.log(7850, math.pow(2, global_epoch + 2)))  ##计算簇数
                    num_in_clusters = math.ceil(len(mean_sum_noise_gradients_shape) / num_clusters)  ##计算平均每簇的数目（向上取整）
                    print("查看服务端分簇数：{}".format(num_clusters))
                    #median_list = [(x * num_in_clusters + int(num_in_clusters / 2)) for x in range(num_clusters)] ##计算每簇中位数的索引（一个小bug)
                    order_gradient_index=np.argsort(np.abs(mean_sum_noise_gradients_shape)) ##将梯度绝对值从大到小排序返回索引
                    gradient_sort = sorted(np.abs(mean_sum_noise_gradients_shape))
                    median_list=[]
                    for i_00 in range(num_clusters-1):
                        median=np.median(gradient_sort[i_00*num_in_clusters:(i_00+1)*num_in_clusters])
                        median_list.append(median)
                    median=np.median(gradient_sort[(num_clusters-1)*num_in_clusters:])
                    median_list.append(median)
                    #print("查看服务器端保存中位数的列表：{}".format(median_list))
                    #(修改一下先将排序后的梯度索引均匀分簇，在返回每簇的中位数。)
                    ##################建立一个保存所有簇的字典 ######################
                    cluster_class={x:[] for x in range(num_clusters)}
                    position_1=bisect.bisect(gradient_sort,median_list[0])
                    cluster_class[0].extend(order_gradient_index[0:position_1])
                    position_2=bisect.bisect(gradient_sort,median_list[num_clusters-1])
                    cluster_class[num_clusters-1].extend(order_gradient_index[position_2:])
                    #######################分簇###################################
                    for i_1 in range (len(gradient_sort[position_1:position_2])):
                        position=bisect.bisect(median_list,gradient_sort[position_1+i_1])
                        if abs(gradient_sort[position_1+i_1]-median_list[position-1])>abs(gradient_sort[position_1+i_1]-median_list[min(position,len(median_list)-1)]):
                            cluster_class[min(position,len(median_list)-1)].append(order_gradient_index[position_1+i_1])
                        else:
                            cluster_class[position-1].append(order_gradient_index[position_1 + i_1])
                    # for i_1 in range(num_clusters-1):
                    #     for i_2 in range (num_in_clusters-1):
                    #         if abs(mean_sum_noise_gradients_shape[order_gradient_index[median_list[i_1]+i_2]]-mean_sum_noise_gradients_shape[order_gradient_index[median_list[i_1]]])>abs(mean_sum_noise_gradients_shape[order_gradient_index[median_list[i_1]+i_2]]-mean_sum_noise_gradients_shape[order_gradient_index[median_list[i_1+1]]]):
                    #             cluster_class[i_1+1].append(order_gradient_index[median_list[i_1]+i_2])
                    #         else:
                    #             cluster_class[i_1].append(order_gradient_index[median_list[i_1]+i_2])
                    ###################去除离散值,返回每个簇的剪切值#################################
                    for i_3 in range(num_clusters):
                        cluster_gradient=[gradient_sort[x] for x in cluster_class[i_3]]
                        #print("查看每一簇的梯度绝度值数目：{}".format(len(cluster_gradient)))
                        clip = every_cluster_clip(cluster_gradient,self.MAD_n)
                        for i_4 in range(len(cluster_class[i_3])):
                            clip_vector[cluster_class[i_3][i_4]]= min(clip,self.clip_C)
                # 步骤六：根据学习率计算更新值

                #self.learning_rate = 0.5 * math.cos(count_iter * 3.14 / 120)
                latest_noise_gradients=self.learning_rate*np.array(mean_sum_noise_gradients_shape)              ###注意学习率的调整
                #步骤七：将聚合后的一维梯度向量变换形状
                new_shape = []
                new_shape.append(np.reshape(latest_noise_gradients[:self.dim_model], (self.dim_x, self.dim_y)))
                new_shape.append(latest_noise_gradients[self.dim_model:])
                #步骤八：更新全局模型参数
                self.latest_model = [u -v for (u, v) in zip(self.latest_model, new_shape)]
                self.client_model.set_params(self.latest_model)
                if self.learning_rate > 0.05:
                    if iter == len(self.clients)-self.clients_per_round:
                        self.learning_rate-=0.01
                print("查看每次迭代的学习率：{}".format(self.learning_rate))
                count_iter += 1

        # final test model
        self.evaluate(count_iter)

    #################################EVALUATING###############################
    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def test(self):
        '''tests self.latest_model on given clients
        '''

        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def evaluate(self, i):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        train_loss = np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])
        train_acc = np.sum(stats_train[3])*1.0/np.sum(stats_train[2])
        test_acc = np.sum(stats[3])*1.0/np.sum(stats[2])

        tqdm.write("===================== round {} =====================".format(i))
        tqdm.write('At round {} training loss: {}'.format(i, train_loss))
        tqdm.write('At round {} training accuracy: {}'.format(i, train_acc))
        tqdm.write('At round {} testing accuracy: {}'.format(i, test_acc))
        self.metrics.accuracies.append(test_acc)
        self.metrics.train_accuracies.append(train_acc)
        self.metrics.train_losses.append(train_loss)
        self.metrics.write()

    #################################LOCAL PROCESS##################################
    def local_process(self, flattened):
        '''
        DO NOTHING
        1. non-private
        2. no clipping
        3. no sparsification
        (for npsgd)
        '''
        return flattened

    #################################AVERAGE/AGGREGATE##############################
    def server_process(self, messages):
        '''
        ONLY AGGREGATE
        weighted or evenly-weighted by num_samples
        '''
        if len(messages) == 1:
            total_weight, base = self.aggregate_e(messages)
        else:
            total_weight, base = self.aggregate_w(messages)
        return self.average(total_weight, base)
    
    def average(self, total_weight, base):
        '''
        total_weight: # of aggregated updates
        base: sum of aggregated updates
        return the average update
        '''
        return [(v.astype(np.float16) / total_weight).astype(np.float16) for v in base]

    def average_cali(self, total_weight, base, clip):
        '''
        total_weight: # of aggregated updates
        base: sum of aggregated updates
        return the average update after transforming back from [0, 1] to [-C, C]
        '''
        return [transform((v.astype(np.float16) / total_weight), 0, 1, -self.clip_C, self.clip_C).astype(np.float16) for v in base]
    
    def aggregate_w(self, wsolns):
        total_weight = 0.0  
        base = [0] * len(wsolns[0][1])
        for w, soln in wsolns:
            total_weight += w
            for i, v in enumerate(soln):
                base[i] = base[i] + w * v.astype(np.float16)
        return total_weight, base

    def aggregate_e(self, solns):
        """

        """
        total_weight = 0.0
        base = [0] * len(solns[0]) #这里的base大小为2，内容就是[0,0]

        for soln in solns: 
            total_weight += 1.0
            for i, v in enumerate(soln):
                base[i] = base[i] + v.astype(np.float16)

        #print("查看base值：{}".format(base[0][0:10]))
        return total_weight, base

    def aggregate_p(self, solns):
        _, base = self.aggregate_e(solns)
        """
        这里的self.choice_list包含的是157*1000的值，np.bincount相当于求取每个维度被用户使用的次数。比如：0维多少个用户上传，1维多少个用户上传。长度一共7850
        """
        m_s = np.bincount(self.choice_list, minlength=(self.dim_model + self.dim_y))
        m_n = np.ones(len(m_s))*self.m_p - m_s #m_n便是一个7850长度的数组，表示有多少需要填充
        #以下语句用来判断是否满足条件
        assert len(np.where(m_n<0)[0]) == 0, 'ERROR: Please choose a larger m_p (smaller mp_rate) and re-run, cause {}>{}'.format(max(m_s), self.m_p)
        dummies = np.zeros(len(m_n))

        sigma = (2*self.clip_C/self.epsilon) * math.sqrt(2 * math.log(1.25/self.delta))
        for i, v in enumerate(m_n):
            assert self.mechanism == 'laplace', "Please use laplace for v1-v3"
            #
            dummies[i] = sum(np.random.laplace(loc=0.5, scale=1.0/self.epsilon, size=int(v))) - 0.5*(self.m_p-self.em_s)
        d_noise = []
        d_noise.append(np.reshape(dummies[:self.dim_model], (self.dim_x, self.dim_y)))
        d_noise.append(dummies[self.dim_model:])

        self.choice_list = []
        return [transform( (v+noise)/self.m_p, 0, 1, -self.clip_C, self.clip_C).astype(np.float16) for v, noise in zip(base, d_noise)]