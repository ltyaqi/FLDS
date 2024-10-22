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

def genarate_clip_gradients(gradients, epsilon, delta, clip_bound):

    main_epsilon = 0.9 * 0.1
    sup_epsilon = 0.1 * 0.1
    clip_bound = 10

    sigma = math.sqrt(2 * math.log(1.25/delta)) / main_epsilon

    sigma_b = sigma / math.sqrt(0.4)
    temp_new = math.pow(sigma, -2) - math.pow(2*sigma_b, -2)
    sigma_new = math.pow(temp_new, -0.5)

    gradients_norm = []
    for gradient_vector in gradients:
        gradient_vector_norm = np.linalg.norm(gradient_vector)
        gradients_norm.append(gradient_vector_norm)

    gradient_dict = dict(zip(gradients_norm, gradients))
    new_gradients = []
    b_k = 0
    for gradient_vector_norm, gradient_vector in gradient_dict.items():
        clip_yes = clip_bound / gradient_vector_norm
        if clip_yes < 1:
            b_k = b_k + 1
            gradient_vector = gradient_vector * clip_yes
        new_gradients.append(gradient_vector)

    noise_sigma = math.pow(sigma_new, 2) * math.pow(clip_bound, 2)
    noise = np.random.normal(loc=0,scale=noise_sigma)
    ones_array = np.ones(7850)
    noise_array = noise * ones_array

    sum_gradients = np.sum(new_gradients, axis=0)
    sum_gradients_noise = sum_gradients + noise_array
    final_gradient = (1/7850) * sum_gradients_noise

    noise_sigma_b = math.pow(sigma_b, 2)
    noise_k = np.random.normal(loc=0,scale=noise_sigma_b)
    change_b_k = b_k + noise_k
    q = 0.5
    temp_exp = -0.01*(change_b_k-q)
    print("noise_sigma: ", noise_sigma)
    print("noise_sigma_b: ", noise_sigma_b)
    print("b_k: ", b_k)
    print("noise_k: ", noise_k)
    print("change_b_k: ", change_b_k)
    print("temp_exp: ", temp_exp)
    new_clip_bound = clip_bound * math.exp(temp_exp)

    return final_gradient, new_clip_bound