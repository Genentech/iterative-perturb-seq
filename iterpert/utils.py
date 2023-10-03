
from .query_strategies import KMeansSampling, kernel_based_active_learning, TypiClust
import os
import pickle

def get_strategy(name):
    
    if name == "KMeansSampling":
        return KMeansSampling
    elif name =="kernel_based_active_learning":
        return kernel_based_active_learning
    elif name == "TypiClust":
        return TypiClust
    else:
        raise NotImplementedError
    

def load_kernel(data_path, kernel_name):            
    
    if not os.path.exists(data_path + kernel_name):
        print(data_path + kernel_name)
        raise ValueError('Kernel does not exist')
    with open(data_path + kernel_name + '/pert_list.pkl', 'rb') as f:
        pert_list = pickle.load(f)
    with open(data_path + kernel_name + '/kernel.pkl', 'rb') as f:
        kernel_npy = pickle.load(f)
    with open(data_path + kernel_name + '/feat.pkl', 'rb') as f:
        feat = pickle.load(f)
    return pert_list, kernel_npy, feat