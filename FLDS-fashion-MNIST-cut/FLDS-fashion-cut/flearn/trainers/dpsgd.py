import numpy as np
from tqdm import tqdm, trange
import math

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import clip
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class Server(BaseFedarated):
    '''
    - one round: one epoch
    - sequentially sample every batch of client for SEVERAL iterations in one round
    - local update is trained with local epoches (--num_epochs) on full-batch
    - evaluate per (--eval_every) iterations

    - full vector aggregation
    '''
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train (dpSGD)')
        self.inner_opt = GradientDescentOptimizer(learning_rate=params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.sigma = (2*self.norm/self.epsilon) * math.sqrt(2 * math.log(1.25/self.delta))
        print("global sigma for gaussian is {}".format(self.sigma))

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened):
        processed_update = clip(flattened, self.norm)  # L_2 clipping
        return processed_update

    def server_process(self, solns):  # DP equals to DDP with the same centralized gaussian noise (privacy level)
        total_weight, base = self.aggregate_e(solns)
        for i, _ in enumerate(base):
            base[i] = base[i] + np.random.normal(0, self.sigma, base[i].shape)
        averaged_soln = self.average(total_weight, base)
        return averaged_soln
