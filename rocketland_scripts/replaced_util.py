import numpy as np
import torch
import os


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def estimate_failure_probability(HJ_value):
    '''
    Input: HJ_value: 2D numpy array(samples, 1), unnormalized HJ value V(T,x)
    
    Output: P(F_s | x), 2D numpy array, failure probability at each state x
    '''
    result = np.zeros(HJ_value.shape)
    
    result[HJ_value > 0] = 1
    
    result[HJ_value <= 0] = sigmoid(HJ_value[HJ_value <= 0])
    
    return result