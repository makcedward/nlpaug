import numpy as np

def standard_norm(data):
    means = data.mean(axis =1)
    stds = data.std(axis= 1, ddof=1)
    data = (data - means[:, np.newaxis]) / stds[:, np.newaxis]
    return np.nan_to_num(data)

def l1_norm(data):
    _norm = np.array([x.sum(axis=0) for x in data])
    data = data/_norm[:, np.newaxis]
    return np.nan_to_num(data)

def l2_norm(data):
    _norm = np.array([np.sqrt((x*x).sum(axis=0)) for x in data])
    data = data/_norm[:, np.newaxis]
    return np.nan_to_num(data)
