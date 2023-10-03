
from query_strategies import KMeansSampling, kernel_based_active_learning, TypiClust

def get_strategy(name):
    
    if name == "KMeansSampling":
        return KMeansSampling
    elif name =="kernel_based_active_learning":
        return kernel_based_active_learning
    elif name == "TypiClust":
        return TypiClust
    else:
        raise NotImplementedError
    