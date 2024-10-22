import numpy as np
import math
import tensorflow as tf
from tqdm import trange, tqdm
import random

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics

from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import transform
from flearn.utils.utils import genarate_clip_cluster


class BaseFedarated(object):
    def __init__(self, params, learner, data):
        for key, val in params.items():
            setattr(self, key, val)

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(
            *params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(data, self.dataset, self.model,
                                          self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()

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
            elif dataset_name == 'mnist':
                return 784*10, 784, 10
        else:
            raise "Unknown dataset and model"

    def setup_clients(self, dataset, dataset_name, model_name, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''

        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(id=u, group=g, dataset_name=dataset_name, model_name=model_name,  # noqa: E501
                              train_data=train_data[u], eval_data=test_data[u], model=model) for u, g in zip(users, groups)]  # noqa: E501
        return all_clients


    #################################TRAINING#################################
    def train_grouping(self):
        count_iter = 0
        clip_vector = [self.norm] * 7850
        for global_epoch in range(self.num_rounds):  # 全体的数据集epoch次数
            # loop through mini-batches of clients
            # for iter in range(0, len(self.clients), self.clients_per_round): #default=1000
            for iter in range(0, len(self.clients), self.clients_per_round):
                print("iter is:{}".format(iter))
                if count_iter % self.eval_every == 0:
                    self.evaluate(count_iter)
                random.shuffle(self.clients)
	selected_clients = self.clients[iter: iter + self.clients_per_round]
                # 新选择的用户数目，这一行可以自己调节
                # new_selected_clients = selected_clients[:1000]
                csolns = []
                for client_id, c in enumerate(selected_clients):
                    # distribute global model
                    c.set_params(self.latest_model)
                    # local iteration on full local batch of client c
                    num_samples, grads = c.get_grads(7850)
                    csolns.append(grads)

                ########################## local process #########################
                # 步骤二：根据随机矩阵从temper当中抽取梯度,存放到choices_gradients当中
                temper = []
                # temper当中最终存放的是1000个clients的梯度，
                for id, csoln in enumerate(csolns):
                    flattened = process_grad(csoln)
                    # 将用户的梯度展平为一维数组，按剪切值向量进行剪切
                    processed_update = self.local_process(flattened, clip_vector)
                    temper.append(processed_update)
                # 步骤四：每个维度进行求和，平均
                sum_gradients_shape = np.sum(temper, axis=0)
                mean_sum_noise = sum_gradients_shape/1000
                if (count_iter == 0 or iter == (len(self.clients) - self.clients_per_round)):
                    clip_vector = genarate_clip_cluster(mean_sum_noise,clip_vector,self.norm)
                #加噪：
                for i in range(len(sum_gradients_shape)):
                    sigma = (2*clip_vector[i]/self.epsilon) * math.sqrt(2 * math.log(1.25/self.delta))
                    sum_gradients_shape[i] = sum_gradients_shape[i] + np.random.normal(loc=0,scale=sigma)
                # 步骤五：服务器端在每次epoch结束的那次迭代中确定剪切向
                mean_sum_noise_gradients_shape =  sum_gradients_shape/1000

                # 步骤六：根据学习率更新梯度
                latest_noise_gradients = self.learning_rate * np.array(mean_sum_noise_gradients_shape)  ###注意学习率的调整
                # 步骤七：将聚合后的一维梯度向量变换形状
                new_shape = []
                new_shape.append(np.reshape(latest_noise_gradients[:self.dim_model], (self.dim_x, self.dim_y)))
                new_shape.append(latest_noise_gradients[self.dim_model:])
                # 步骤八：更新全局模型参数
                self.latest_model = [u - v for (u, v) in zip(self.latest_model, new_shape)]
                self.client_model.set_params(self.latest_model)
                # 更新学习率
                # if iter == len(self.clients) - self.clients_per_round:
                #     if self.learning_rate > 0.05:
                #         self.learning_rate -= 0.01
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
                base[i] = base[i] + w * v.astype(np.float32)
        return total_weight, base

    def aggregate_e(self, solns):
        total_weight = 0.0
        base = [0] * len(solns[0])
        for soln in solns: 
            total_weight += 1.0
            for i, v in enumerate(soln):
                base[i] = base[i] + v.astype(np.float32)
        return total_weight, base    

    def aggregate_p(self, solns):
        _, base = self.aggregate_e(solns)
        m_s = np.bincount(self.choice_list, minlength=(self.dim_model + self.dim_y))
        m_n = np.ones(len(m_s))*self.m_p - m_s
        assert len(np.where(m_n<0)[0]) == 0, 'ERROR: Please choose a larger m_p (smaller mp_rate) and re-run, cause {}>{}'.format(max(m_s), self.m_p)
        dummies = np.zeros(len(m_n))

        sigma = (2*self.clip_C/self.epsilon) * math.sqrt(2 * math.log(1.25/self.delta))
        for i, v in enumerate(m_n):
            assert self.mechanism == 'laplace', "Please use laplace for v1-v3"
            dummies[i] = sum(np.random.laplace(loc=0.5, scale=1.0/self.epsilon, size=int(v))) - 0.5*(self.m_p-self.em_s)
        d_noise = []
        d_noise.append(np.reshape(dummies[:self.dim_model], (self.dim_x, self.dim_y)))
        d_noise.append(dummies[self.dim_model:])

        self.choice_list = []  # empty the choise list after each aggregation
        return [transform( (v+noise)/self.em_s, 0, 1, -self.clip_C, self.clip_C).astype(np.float16) for v, noise in zip(base, d_noise)]