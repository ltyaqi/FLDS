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
# import numpy as np
# np.set_printoptions(threshold=np.inf)


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
        for i in range(self.num_rounds): #全体的数据集epoch次数

            # loop through mini-batches of clients
            #for iter in range(0, len(self.clients), self.clients_per_round): #default=1000
            for iter in range(0, len(self.clients), 1000):
                if count_iter % self.eval_every == 0:
                    self.evaluate(count_iter)
                random.shuffle(self.clients)
	selected_clients = self.clients[iter: iter + self.clients_per_round]
                # print(len(selected_clients))
                #新选择的用户数目，这一行可以自己调节
                #new_selected_clients = selected_clients[:1000]
                csolns = []
                for client_id, c in enumerate(selected_clients):
                    # distribute global model
                    c.set_params(self.latest_model)
                    # local iteration on full local batch of client c
                    num_samples, grads = c.get_grads(7850)
                    # if client_id in [50,100,999] :
                    #     print("查看第{}个用户的梯度前10维：{}".format(client_id,grads[:10]))
                    #     print("查看第{}个用户的梯度第1200到1210维：{}".format(client_id,grads[1200:1210]))
                    # track computational cost
                    #self.metrics.update(rnd=i, cid=c.id, stats=stats)
                    # local update
                    #model_updates = [u - v for (u, v) in zip(soln[1], self.latest_model)]
                    # aggregate local update
                    csolns.append(grads)
                """
                ########################## local updating ##############################
                for client_id, c in enumerate(new_selected_clients): #这里的client_id没使用
                    # distribute global model
                    c.set_params(self.latest_model) #为当前用户更新到新的模参
                    # local iteration on full local batch of client c
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)#每个用户开始本地训练
                    # track computational cost
                    self.metrics.update(rnd=i, cid=c.id, stats=stats) #记录参数状态，训练结果
                    # local update（这里指的是本地梯度）
                    model_updates = [u - v for (u, v) in zip(soln[1], self.latest_model)]
                    # aggregate local update
                    csolns.append(model_updates)
                """
                # ######################### local process #########################
                #-------------------------------当前方案上界------------------------------------
                sum_sample_clients = 100
                #步骤一：生成随机矩阵和噪音矩阵
                random_index_matrix, noise_matrix = generater_random_matrix(1000, 7850, 100, 785, self.epsilon,self.clip_C)
                #步骤二：根据随机矩阵从temper当中抽取梯度,存放到choices_gradients当中
                temper = [] #temper当中最终存放的是1000个clients的梯度，
                for csoln in csolns:
                    flattened = np.clip(csoln,-self.clip_C,self.clip_C)         #将用户的梯度展平为一维数组
                    temper.append(flattened)
                #扰动前求和取平均，只看前150维
                # test_result = np.sum(temper,axis=0) / 1000
                # test_result.tolist()
                #print("查看用户端扰动前，110维~130维：{}".format(test_result[110:130]))

                #clip_gradient = np.clip(temper,-self.clip_C,self.clip_C) #梯度剪切
                choices_gradients = np.where(random_index_matrix == 1,temper,0) #选择相应的梯度并且归一化处理，其他置为0

                #步骤三：给选择的梯度添加噪音扰动
                noise_choices_gradients = new_randomizer(choices_gradients,noise_matrix)
                #步骤四：每个维度进行求和，平均
                mean_sum_noise_gradients_shape = np.sum(noise_choices_gradients,axis=0) / sum_sample_clients
                # print("查看平均梯度的0到30维:{}".format(mean_sum_noise_gradients_shape[0:30]))
                # print("查看平均梯度的500到530维:{}".format(mean_sum_noise_gradients_shape[500:530]))
                # print("查看平均梯度的1000到1030维:{}".format(mean_sum_noise_gradients_shape[1000:1030]))
                # print("查看平均梯度的2000到2030维:{}".format(mean_sum_noise_gradients_shape[2000:2030]))
                # print("查看平均梯度的2000到2030维:{}".format(mean_sum_noise_gradients_shape[2000:2030]))

                # b = list(np.argsort(np.abs(mean_sum_noise_gradients_shape)))
                # count_output = 0
                # for j in b:#
                #     print(j, end=', ')
                #     count_output += 1  # 开始计数
                #     if count_output % 30 == 0:  # 每10个换行
                #         print(end='\n')


                #步骤五：梯度剪切变换归一化处理
                ##不需要变换归一化了
                # latest_noise_gradients = [transform(v,0,1,-self.clip_C,self.clip_C).astype(np.float16) for v in mean_sum_noise_gradients_shape]
                #print("查看服务器端，扰动后110维~130维：{}".format(latest_noise_gradients[110:130]))
                
                # 步骤六：根据学习率计算更新值
                #0.5 * math.cos(count_iter * 3.14 / 120)
                latest_gradients=self.learning_rate*np.array(mean_sum_noise_gradients_shape)              ###注意学习率的调整
                #latest_gradients=self.learning_rate*np.array(mean_sum_noise_gradients_shape)
                #步骤七：将聚合后的一维梯度向量变换形状
                new_shape = []
                new_shape.append(np.reshape(latest_gradients[:self.dim_model], (self.dim_x, self.dim_y)))
                new_shape.append(latest_gradients[self.dim_model:])
                #步骤八：更新全局模型参数
                self.latest_model = [u -v for (u, v) in zip(self.latest_model, new_shape)]
                self.client_model.set_params(self.latest_model)
                count_iter += 1
                # if iter == len(self.clients)-self.clients_per_round:
                #     self.learning_rate-=0.01
                # print("查看每次迭代的学习率：{}".format(self.learning_rate))
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