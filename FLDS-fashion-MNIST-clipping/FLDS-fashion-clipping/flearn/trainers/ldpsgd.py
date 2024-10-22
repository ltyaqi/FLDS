import numpy as np
from tqdm import tqdm, trange
import math

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import clip, sparsify, transform
from flearn.utils.priv_utils import sampling_randomizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class Server(BaseFedarated):
    '''
    traditional LDP-FL
    1. sampling depends on self.rate
    2. gaussian distribution for (epsilon, delta_lk)-LDP(DP) privacy
    '''
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train (LDP-FL)')
        self.inner_opt = GradientDescentOptimizer(learning_rate=params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.clip_C = self.norm
        self.sample = int( (self.dim_model + self.dim_y)/self.rate)
        self.eps_ld = self.epsilon / self.sample

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened):
        choices = np.random.choice(flattened.size, self.sample)
        return sampling_randomizer(flattened, choices, self.clip_C, self.eps_ld, self.delta, self.mechanism)

    def server_process(self, messages):
        '''
        basic aggregate, scale with rate when Top-k is applied (when rate > 1)
        '''
        total_weight, base = self.aggregate_e(messages)
        return self.average_cali(total_weight/self.rate, base, self.clip_C)
