import pickle
import random

import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def iid_divide(l, g):
    '''
    divide list l among g groups
    each group has either int(len(l)/g) or int(len(l)/g)+1 elements
    returns a list of groups
    '''
    num_elems = len(l)
    group_size = int(len(l)/g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size*i:group_size*(i+1)])
    bi = group_size*num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi+group_size*i:bi+group_size*(i+1)])
    return glist

def sparsify(updates, topk):
    '''
    return sparsified updates, with non-top-k as zeros
    '''
    d = updates.size
    non_top_idx = np.argsort(np.abs(updates))[:d-topk]
    updates[non_top_idx] = 0
    return updates

def topindex(updates, topk):
    '''
    return top=k indexes
    '''
    d = updates.size
    return np.argsort(np.abs(updates))[d-topk:]       ##按从小到大排序，然后输出对应的索引

def clip(updates, threshold):
    '''
    clip updates vector with L2 norm threshold
    input
        updates: 1-D vector
        threshold: L2 norm
    
    return:
        clipped 1-D vector
    '''

    # L2 norm
    L2_norm = np.linalg.norm(updates, 2)
    if L2_norm > threshold:
        updates = updates * (threshold * 1.0 / L2_norm)

    # # threshold for each dimension
    # updates = np.clip(updates, -threshold, threshold)
    return updates

def discrete(x, b):
    xk = np.floor(v*b)
    r = np.random.rand()
    if r < (x*k - xk):
        return xk + 1
    else:
        return xk

def shape_back(flattened_queried):
    queried_weights = []
    queried_weights.append(np.reshape(flattened_queried[:7840], (784, 10)))
    queried_weights.append(flattened_queried[7840:])
    return queried_weights


def transform(v, left, right, new_left, new_right):
    '''
    transform a vector/value from [left, right] to [new_left, new_right]
    '''
    return new_left + (new_right - new_left)*(v - left)/(right - left)


def one_laplace(eps, sensitivity=1):#,count_noist
    '''
    sample a laplacian noise for a scalar
    '''
    np.random.seed()#count_noist)
    return np.random.laplace(loc=0, scale=sensitivity/eps)
def every_cluster_clip(data,n):
    print("查看传入函数的数据的数目：{}".format(len(data)))
    median = np.median(data)  # 中位数
    deviations = abs(data - median)
    mad = np.median(deviations)
    remove_idx = np.where(abs(data - median) > n * mad)
    new_data = np.delete(data, remove_idx)
    print("查看去除离群值后的值数目：{}".format(len(new_data )))
    clip=np.max(new_data)
    return clip



def generater_random_matrix(clients,dimensions,sum_sample_clients,sum_sample_dimensions):
    """
    算法步骤：
    1.先生成单元矩阵，满足每一行都是157，每一列都是1
        1.1 先生成50个全0矩阵，形状为(157,50)
        1.2 分别使得第i列的值为1
        1.3 恢复每个矩阵的形状为(7850,)
    2.然后复制此单元矩阵20份
    3.打乱行与列
    """
    # 先生成当前批次的空矩阵，50个.每一个都是二维矩阵，形式都是(157,50)
    sum_sample_dimensions_of_total_dimensions = sum_sample_dimensions / dimensions
    purpose_clients = sum_sample_dimensions_of_total_dimensions * sum_sample_clients

    unit_matrixes = []
    for i in range(int(1 / sum_sample_dimensions_of_total_dimensions)):
        unit_matrixes.append(np.zeros((sum_sample_dimensions,int(1 / sum_sample_dimensions_of_total_dimensions))))

    #分别使得第i列的值为1
    count = 0
    new_unit_matrixes = [] #new_unit_matrixes就是满足条件的单位矩阵
    for unit_matrix in unit_matrixes:
        unit_matrix[:,count] = 1
        count += 1
        new_unit_matrixes.append(np.reshape(unit_matrix,7850))

    #扩展单元矩阵20份
    random_choices_matrixes = []
    for i in range(sum_sample_clients):
        for new_unit_matrix in new_unit_matrixes:
            random_choices_matrixes.append(new_unit_matrix)

    #分别进行行与列的打乱

    random_choices_matrixes = np.array(random_choices_matrixes)
    np.random.seed()
    random_choices_matrixes = random_choices_matrixes[np.random.choice(clients,clients,replace=False),:]
    np.random.seed()
    random_choices_matrixes = random_choices_matrixes[:,np.random.choice(dimensions,dimensions,replace=False)]

    """
    #-----------------------------------------------生成噪音矩阵
    #首先生成相同形状的空矩阵
    noise_matrix = np.zeros((clients,dimensions))
    #count_noist = count_iter
    for i in range(random_choices_matrixes.shape[0]):
        for j in range(random_choices_matrixes.shape[1]):
            if random_choices_matrixes[i][j] == 1:
                np.random.seed()
                noise_matrix[i][j] = one_laplace(epsilon,sensitivity=1)
    #查看矩阵的值
    #print("查看噪音矩阵的值：{}".format(np.sum(noise_matrix,axis=0) / 1000)[:150])
    """
    return random_choices_matrixes




def image(list):  # 对二维列表每一行画出频数直方图
    plt.figure()  #初始化一张图
    plt.hist(list, bins=100, range=(-0.5, 0.5))  # 直方图关键操作
    plt.grid(alpha=0.5,linestyle='-.')  # 网格线
    plt.xlabel('Gradients')
    plt.ylabel('Number of Events')
    plt.title(r'Distribution of Gradients')
    plt.show()

def expMechanism(scores,eps,sensitivity):
    probability0 = []
    probability1 = []
    x=0
    for score in scores:
        probability0.append(np.exp(0.5*eps*score/sensitivity))
    sum = np.sum(probability0)
    # 归一化处理
    for i in range(len(probability0)):
        probability1.append(probability0[i]/sum)
    ret = random.random()
    for j in range(len(scores)):
        sum += probability1[j]
        if sum >= ret:
            x=j
            break
    return x

