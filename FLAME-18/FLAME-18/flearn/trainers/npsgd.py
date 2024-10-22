import numpy as np
from tqdm import tqdm

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import sparsify
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class Server(BaseFedarated):
    '''
    - one round: one epoch
    - sequentially sample every batch of client for SEVERAL iterations in one round # noqa: E501
    - local update is trained with local epoches (--num_epochs) on full-batch
    - evaluate per (--eval_every) iterations

    - full vector aggregation
    '''

    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train (npSGD)')
        self.inner_opt = GradientDescentOptimizer(learning_rate=params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        if self.rate > 1:
            self.topk = int( (self.dim_model + self.dim_y)/self.rate)
            print("Topk selecting {} dimensions".format(self.topk))

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened):
        '''
        if sparsification is required (self.rate >1) for non-private version, call sparsify function
        else return the raw vector (save sorting costs)
        '''
        if self.rate > 1:
            return(sparsify(flattened, self.topk))
        else:
            return flattened

    def server_process(self, messages):
        '''
        basic aggregate, but enlarge the learning rate when Top-k is applied
        '''
        total_weight, base = self.aggregate_e(messages)
        return self.average(total_weight/self.rate, base)