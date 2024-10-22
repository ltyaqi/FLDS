import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

# PKGS: tensorflow 1.3

# GLOBAL PARAMETERS
OPTIMIZERS = ['npsgd', 'dpsgd', 'ldpsgd', 'v1sgd', 'v2sgd', 'v3sgd']
DATASETS = ['mnist']

MODEL_PARAMS = {
    'mnist.mclr': (10,), # num_classes
    'mnist_cpsgd.mclr': (10,), # num_classes
    'shakespeare.stacked_lstm': (80, 80, 50), # seq_len, emb_dim, num_hidden
    'adult.mclr': (2,), # num_classes
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()
    # main setting
    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='v3sgd')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='mnist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='mclr')
    # initialization global epoch, client batchs
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=10) #
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=1000)
    # for local update
    parser.add_argument('--batch_size',    # LOCAL: no greater than the local data size
                        help='batch size for local iteration (for sampling-based, denotes the number of local data that will be used throughout one epoch, for grouping-based, denotes the batch size for one/multiple local iterations for one updating);',
                        type=int,
                        default=7)
    parser.add_argument('--num_epochs',    # LOCAL: local epoch
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=10)
    # for global model
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    # for privacy
    parser.add_argument('--epsilon',
                        help='eps_c for DP, LDP, eps_lk/ld for SS',
                        type=float,
                        default=0.5)
    parser.add_argument('--delta',
                        help='delta for DP, delta_lk for LDP(no SS-FL)',
                        type=float,
                        default=0.001)
    parser.add_argument('--mechanism',
                        help='type of local randomizer: gaussian, laplace, krr',
                        type=str,
                        default='gaussian')
    # for sparsification
    parser.add_argument('--norm',
                        help='L2 norm clipping threshold',
                        type=float,
                        default=10)
    parser.add_argument('--rate',
                        help='compression rate, 1 for no compression',
                        type=int,
                        default=1)
    
    # for padding
    parser.add_argument('--mp_rate',
                        help='under factor for mp=m/mp_rate',
                        type=float,
                        default=1)
    parser.add_argument('--MAD_n',
                        help='number of n to MAD;',
                        type=float,
                        default=3)


    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])

    # load selected model
    model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data
    path = "/".join(os.path.abspath(__file__).split('/')[:-1])
    log_path = os.path.join(os.path.abspath('.'), 'out_new', options['dataset']) 
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    train_path = os.path.join(path, 'data/train')
    test_path = os.path.join(path, 'data/test')
    dataset = read_data(train_path, test_path)

    # call trainer
    t = optimizer(options, learner, dataset)
    t.train()
    
if __name__ == '__main__':
    main()
