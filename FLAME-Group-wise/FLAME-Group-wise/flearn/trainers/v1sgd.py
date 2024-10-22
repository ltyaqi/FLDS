import numpy as np
from tqdm import tqdm, trange
import math
from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import clip, sparsify
from flearn.utils.priv_utils import full_randomizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class Server(BaseFedarated):
    '''
    SS-FL-V1
    '''
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train (SS-Simple)')
        self.inner_opt = GradientDescentOptimizer(learning_rate=params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.clip_C = self.norm

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened):
        return full_randomizer(flattened, self.clip_C, self.epsilon, self.delta, self.mechanism)

    def server_process(self, messages):
        '''
        1. average aggregated updates
        2. scale the average back from [0, 1] to [-C, C]
        '''
        total_weight, base = self.aggregate_e(messages)
        return self.average_cali(total_weight, base, self.clip_C)
