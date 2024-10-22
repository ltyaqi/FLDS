import numpy as np
import random
import json
import os
from tqdm import trange
from sklearn.datasets import fetch_openml
random.seed(7)

# Setup directory for train/test data
root = 'data/'
train_file = root + 'train/train.json'
test_file = root + 'test/test.json'
dir_path = os.path.dirname(train_file)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_file)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
mnist = fetch_openml('mnist_784', version=1, data_home=root)
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
max_ = np.max(mnist.data.astype(np.float32))
mnist.data = (mnist.data.astype(np.float32)-mu)/(sigma+0.001)

mnist_data = []
for i in range(10):
    idx = mnist.target==str(i)
    mnist_data.append(mnist.data[idx])

print([len(v) for v in mnist_data])

NUM_USERS = 6000
SAMPLES_PER_USER=10
NUM_CLASSES = 10
TRAINT_RATE = 0.7

###### CREATE USER DATA SPLIT #######
print("Assign samples to each user")
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(NUM_CLASSES, dtype=np.int64)
for user in range(NUM_USERS):
    for j in range(SAMPLES_PER_USER):
        l = (user+j) % NUM_CLASSES
        X[user].append(mnist_data[l][idx[l]].tolist())
        y[user].append(np.array(l).tolist())
        idx[l] += 1
print("idx=", idx)


print("Create data structure")
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

print("Setup users")
for i in trange(NUM_USERS, ncols=120):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(TRAINT_RATE*num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("writing...")
    
with open(train_file, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_file, 'w') as outfile:
    json.dump(test_data, outfile)
