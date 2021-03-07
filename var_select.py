import numpy as np


def kfilter(W, offset = 0.0, q = 0.1):
    t = np.insert(np.abs(W[W != 0]), 0, 0) # omitting 0 value and then add zero at the begining
    t = np.sort(t)
    ratio = np.zeros(len(t));
    for i in range(len(t)):
        ratio[i] = (offset + np.sum(W <= -t[i])) / np.maximum(1.0, np.sum(W >= t[i]))
        
    index = np.where(ratio <= q)[0]
    if len(index)== 0:
        thresh = float('inf')
    else:
        thresh = t[index[0]]
       
    return thresh

def select(W, offset = 0.0, nominal_fdr = 0.1):
    
    W_threshold = kfilter(W, q = nominal_fdr)
    selected = np.where(W >= W_threshold)[0]
    return selected, W_threshold