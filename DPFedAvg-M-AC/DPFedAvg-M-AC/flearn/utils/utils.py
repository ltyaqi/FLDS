import pickle
import numpy as np
import math
import bisect
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
    return np.argsort(np.abs(updates))[d-topk:]

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

def genarate_clip_cluster(mean_sum_noise_gradients_shape, clip_vector,c):
    thrould_t = np.std(mean_sum_noise_gradients_shape)#求向量标准差
    center_index = [] #中心点的索引
    clsuster_dict = {} #分簇索引
    rng = np.random.default_rng()
    select_index = rng.integers(
        low=0,
        high=len(mean_sum_noise_gradients_shape)
    )
    clsuster_dict[0] = []
    clsuster_dict[0].append(select_index)
    center_index.append(select_index)
    for i in range(len(mean_sum_noise_gradients_shape)):
        flag = False
        if i == select_index:
            continue
        for j in center_index:
            if abs(mean_sum_noise_gradients_shape[i] - mean_sum_noise_gradients_shape[j]) < thrould_t:
                clsuster_dict[center_index.index(j)].append(i)
                flag = True
                break
        if not flag:
            clsuster_dict[len(center_index)] = []
            clsuster_dict[len(center_index)].append(i)
            center_index.append(i)
    for key, value in clsuster_dict.items():
        cluster_gra = [mean_sum_noise_gradients_shape[x] for x in value]
        clip = clip_bound(cluster_gra,c)
        for i in value:
            clip_vector[i] =abs(clip)
    return clip_vector


def clip_bound(cluster,c):
    if len(cluster)==1:
        return c
    B = 0
    for i in cluster[1:]:
        B += i / (len(cluster)-1)
    clip_bound = abs(B)
    # clip_bound = abs(B / (len(cluster) - 1))
    return clip_bound