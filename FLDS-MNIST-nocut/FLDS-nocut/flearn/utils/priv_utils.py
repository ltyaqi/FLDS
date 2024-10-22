import pickle
import numpy as np
import math
import sympy as sp
from flearn.utils.utils import transform, discrete


#################################ADD NOISE#######################################
def add_laplace(updates, sensitivity, epsilon):
    '''
    inject laplacian noise to a vector
    '''
    lambda_ = sensitivity * 1.0 / epsilon
    updates += np.random.laplace(loc=0, scale=lambda_, size=updates.shape)
    return updates

def add_gaussian(updates, eps, delta, sensitivity):
    '''
    inject gaussian noise to a vector
    '''
    sigma = (sensitivity/eps) * math.sqrt(2 * math.log(1.25/delta))
    updates += np.random.normal(0, sigma)
    return updates

def one_gaussian(eps, delta, sensitivity):
    '''
    sample a gaussian noise for a scalar
    '''
    sigma = (sensitivity/eps) * math.sqrt(2 * math.log(1.25/delta))
    return np.random.normal(0, sigma)

def one_laplace(eps, sensitivity):
    '''
    sample a laplacian noise for a scalar
    '''
    return np.random.laplace(loc=0, scale=sensitivity/eps)

def full_randomizer(vector, clip_C, eps, delta, mechanism, left=0, right=1):
    clipped = np.clip(vector, -clip_C, clip_C)
    normalized_updates = transform(clipped, -clip_C, clip_C, left, right)
    if mechanism == 'gaussian':
        perturbed = add_gaussian(normalized_updates, eps, delta, sensitivity=right-left)
    elif mechanism == 'laplace':
        perturbed = add_laplace(normalized_updates, sensitivity=1, epsilon=eps)
    return perturbed


def sampling_randomizer(vector, choices, clip_C, eps, delta, mechanism, left=0, right=1):
    vector = np.clip(vector, -clip_C, clip_C)
    for i, v in enumerate(vector):
        if i in choices:
            normalize_v = transform(vector[i], -clip_C, clip_C, left, right)
            if mechanism == 'gaussian':
                vector[i] = normalize_v + one_gaussian(eps, delta, right-left)
            elif mechanism == 'laplace':
                vector[i] = normalize_v + one_laplace(eps, right-left)
        else:
            vector[i] = 0
    return vector

# def new_randomizer(gradients,cli_C,epsilon,left=0,right=1):
#     #梯度剪切
#     clip_gradient = np.clip(gradients,-cli_C,cli_C)
#     return np.where(clip_gradient == 0,0,clip_gradient+one_laplace(epsilon,right-left))

def latest_new_randomizer(gradients,random_index_matrix,cli_C,epsilon,left=0,right=1):
    clip_gradient = np.clip(gradients,-cli_C,cli_C) #梯度剪切
    return np.where(random_index_matrix == 0,0,clip_gradient+one_laplace(epsilon,right-left))

def new_randomizer(gradients,zero_matrix):
    return gradients+zero_matrix

