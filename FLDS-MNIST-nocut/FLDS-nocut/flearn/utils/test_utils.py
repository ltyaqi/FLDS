import pickle
import random

import numpy as np
import math
import sympy as sp


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
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i:group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def sparsify(updates, topk):
    '''
    return sparsified updates, with non-top-k as zeros
    '''
    d = updates.size
    non_top_idx = np.argsort(np.abs(updates))[:d - topk]
    updates[non_top_idx] = 0
    return updates


def topindex(updates, topk):
    '''
    return top=k indexes
    '''
    d = updates.size
    return np.argsort(np.abs(updates))[d - topk:]


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
    xk = np.floor(v * b)
    r = np.random.rand()
    if r < (x * k - xk):
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
    return new_left + (new_right - new_left) * (v - left) / (right - left)


def one_laplace(eps, sensitivity=1):
    '''
    sample a laplacian noise for a scalar
    '''
    random.seed()
    return np.random.laplace(loc=0, scale=sensitivity / eps)


"""
def generater_random_matrix(clients,dimensions,sum_sample_clients,sum_sample_dimensions,epsilon):

    1.先生成单元矩阵，满足每一行都是157，每一列都是1
        1.1 先生成50个全0矩阵，形状为(157,50)
        1.2 分别使得第i列的值为1
        1.3 恢复每个矩阵的形状为(7850,)
    2.然后复制此单元矩阵20份
    3.打乱行与列

    # 先生成当前批次的空矩阵，50个.每一个都是二维矩阵，形式都是(157,50)
    sum_sample_dimensions_of_total_dimensions = sum_sample_dimensions / dimensions
    purpose_clients = sum_sample_dimensions_of_total_dimensions * sum_sample_clients

    unit_matrixes = []
    for i in range(int(1 / sum_sample_dimensions_of_total_dimensions)):
        unit_matrixes.append(np.zeros((sum_sample_dimensions,int(1 / sum_sample_dimensions_of_total_dimensions))))

    #分别使得第i列的值为1
    count= 0
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

    #-----------------------------------------------生成噪音矩阵
    #首先生成相同形状的空矩阵
    noise_matrix = np.zeros((clients,dimensions))

    for i in range(random_choices_matrixes.shape[0]):
        for j in range(random_choices_matrixes.shape[1]):
            if random_choices_matrixes[i][j] == 1:
                np.random.seed()

                noise_matrix[i][j] = one_laplace(epsilon,sensitivity=1)
    #查看矩阵的值
    #print("查看噪音矩阵的值：{}".format(np.sum(noise_matrix,axis=0) / 1000)[:150])

    return random_choices_matrixes,noise_matrix

"""

"""
def generater_random_matrix(clients,dimensions,epsilon):

    # 先生成当前批次的空矩阵，50个.每一个都是二维矩阵，形式都是(157,50)
    #sum_sample_dimensions_of_total_dimensions = sum_sample_dimensions / dimensions
    #purpose_clients = sum_sample_dimensions_of_total_dimensions * sum_sample_clients

    unit_matrixes = []
    for i in range(40):
        unit_matrixes.append(np.zeros((157,40)))

    #分别使得第i列的值为1
    count = 0
    new_unit_matrixes = [] #new_unit_matrixes就是满足条件的单位矩阵
    for unit_matrix in unit_matrixes:
        unit_matrix[:,count] = 1
        count += 1
        unit_matrix=np.reshape(unit_matrix, 6280)
        unit_matrix=np.append(unit_matrix,[0]*1570)
        new_unit_matrixes.append(unit_matrix)


    #扩展单元矩阵20份
    random_choices_matrixes = []
    for i in range(25):
        for new_unit_matrix in new_unit_matrixes:
            random_choices_matrixes.append(new_unit_matrix)

    #分别进行行与列的打乱
    random_choices_matrixes = np.array(random_choices_matrixes)
    np.random.seed()
    random_choices_matrixes = random_choices_matrixes[np.random.choice(clients,clients,replace=False),:]
    np.random.seed()
    random_choices_matrixes = random_choices_matrixes[:,np.random.choice(dimensions,dimensions,replace=False)]


    #-----------------------------------------------生成噪音矩阵
    #首先生成相同形状的空矩阵
    noise_matrix = np.zeros((1000,7850))
    #
    for i in range(random_choices_matrixes.shape[0]):
        for j in range(random_choices_matrixes.shape[1]):
            if random_choices_matrixes[i][j] == 1:
                np.random.seed()
                noise_matrix[i][j] = one_laplace(epsilon,sensitivity=1)
    #查看矩阵的值
    #print("查看噪音矩阵的值：{}".format(np.sum(noise_matrix,axis=0) / 1000)[:150])

    return random_choices_matrixes,noise_matrix

"""


def generater_random_matrix(clients, dimensions, total_udate_D, eveclient_update_D, each_D_clients, epsilon,
                            zero_index):
    # total_udate_D表示选择更新的总维度，eveclient_update_D表示每个用户更新的维度，each_D_clients每一维上传的用户数，zero_index表示梯度为0的索引
    # print("zero_index:{}".format(zero_index))
    """
    第一步：生成一个dimensions长的全0数组
           按每个用户上传多少维数据对其初始化
    """
    init_list = np.zeros(dimensions)
    for i in range(int(eveclient_update_D)):
        init_list[i * int(total_udate_D / eveclient_update_D)] = 1
    # print("check_init_list:{}".format(init_list))
    # print("check_init_list:{}".format(np.shape(init_list)))
    # print("check_init_list[50]:{}".format(init_list[50]))
    """
    第二步：将数组循环右移，并保存，使得每一维都有数据
    """
    count = 0
    new_unit_matrixes = []
    for _ in range(int(total_udate_D / eveclient_update_D)):
        new_unit_matrixes.append(np.roll(init_list, count))
        count += 1
    # print("new_unit_matrixes_shape:{}".format(np.shape(np.array(new_unit_matrixes))))
    """
    第三步：按每个维度有多少用户上传，将第2步生成的矩阵复制相应份数
    """

    random_choices_matrixes = []
    for i in range(int(each_D_clients)):
        for new_unit_matrix in new_unit_matrixes:
            random_choices_matrixes.append(new_unit_matrix)

    # 分别进行行与列的打乱
    random_choices_matrixes = np.array(random_choices_matrixes)
    # print("check_shape:{}".format(np.shape(random_choices_matrixes)))
    np.random.seed()
    random_choices_matrixes = random_choices_matrixes[np.random.choice(clients, clients, replace=False), :]
    np.random.seed()
    random_choices_matrixes = random_choices_matrixes[:, np.random.choice(dimensions, dimensions, replace=False)]
    real_zero_index = []
    """
    找到打乱后矩阵，列全为0的索引
    """
    for _ in range(dimensions):
        if all(random_choices_matrixes[:, _] == [0] * clients):
            # print("check _:{}".format(_))
            # print("zero_index:{}".format(zero_index))
            real_zero_index.append(_)
    print("check_real_zero_index:{}".format(real_zero_index))

    """
    删除打乱后，列全为0与zero_index中相同的值
    """
    for j in real_zero_index:
        # print("check_j:{}".format(j))
        if j in zero_index:
            real_zero_index.remove(j)
            zero_index.remove(j)
    """
    按上一步求得的两个索引列表的值，在打乱后的矩阵中交换对应列，从而得到指定的随机矩阵
    """
    for i in range(len(zero_index)):
        tmp = np.copy(random_choices_matrixes[:, real_zero_index[i]])
        random_choices_matrixes[:, real_zero_index[i]] = random_choices_matrixes[:, zero_index[:, i]]
        random_choices_matrixes[:, zero_index[:, i]] = tmp

    # -----------------------------------------------生成噪音矩阵
    # 首先生成相同形状的空矩阵
    noise_matrix = np.zeros((clients, dimensions))
    #
    for i in range(random_choices_matrixes.shape[0]):
        for j in range(random_choices_matrixes.shape[1]):
            if random_choices_matrixes[i][j] == 1:
                np.random.seed()
                noise_matrix[i][j] = one_laplace(epsilon, sensitivity=1)
    # 查看矩阵的值
    # print("查看噪音矩阵的值：{}".format(np.sum(noise_matrix,axis=0) / 1000)[:150])

    return random_choices_matrixes, noise_matrix