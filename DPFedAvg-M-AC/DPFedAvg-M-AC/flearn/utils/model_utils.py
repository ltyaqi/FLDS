import json
import numpy as np
import os
import tensorflow as tf

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth=True


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def gen_batch(data, batch_size, num_iter):
    '''
    :params:
        data: data['x'] for features and data['y'] for labels
        batch_size: batch size for local SGD iteration (only once local update)
        num_iter: the expected times that the gen_batch will be called, 
    '''

    data_x = data['x']
    data_y = data['y']
    index = len(data_y)
    assert batch_size <= index, 'Please make sure batch_size_{} is no greater than local dataset_{}'.format(  # noqa: E501
        batch_size, index)

    for i in range(num_iter):
        index += batch_size
        # if the right index (index+batch_size) WILL get out of range, index restart from 0
        if index + batch_size >= len(data_y):  
            index = 0
            np.random.seed(i + 1)
            # randomly shuffle the data after one pass of the entire local set  # noqa: E501
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

        batched_x = data_x[index: index + batch_size]
        batched_y = data_y[index: index + batch_size]

        yield (batched_x, batched_y)


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))
    num_train_per_user = len(train_data[clients[0]]['x'])
    num_test_per_user = len(test_data[clients[0]]['x'])
    print("Each user has {} records for training, {} for testing".format(
        num_train_per_user, num_test_per_user))

    return clients, groups, train_data, test_data


class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []
        self.train_losses = []
        self.path = './out_new'

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        '''write existing history records into a json file'''
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['optimizer'] = self.params['optimizer']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['epsilon'] = self.params['epsilon']
        metrics['delta'] = self.params['delta']
        metrics['norm'] = self.params['rate']
        metrics['rate'] = self.params['rate']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['train_losses'] = self.train_losses
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join(self.path, self.params['dataset'],
                                   'metrics_{}_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(self.params['dataset'],  # noqa: E501
                                                                              self.params['optimizer'],  # noqa: E501
                                                                              self.params['learning_rate'],  # noqa: E501
                                                                              self.params['epsilon'],  # noqa: E501
                                                                              self.params['delta'],  # noqa: E501
                                                                              self.params['norm'],  # noqa: E501
                                                                              self.params['rate'],  # noqa: E501
                                                                              self.params['mp_rate'],
                                                                              self.params['mechanism']))  # noqa: E501
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)
