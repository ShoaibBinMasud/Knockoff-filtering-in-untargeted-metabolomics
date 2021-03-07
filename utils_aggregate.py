import numpy as np
from sklearn.utils import check_random_state


def generate_seedlist(no_bootstraps, random_state):
    #generate seed list to implement bootstrap
    '''
    return a seedlist with no_bootstraps elements
    '''
    if isinstance(random_state, (int, np.int32, np.int64)):
        rng = check_random_state(random_state)
    elif random_state is None:
        rng = check_random_state(0)
    else:
        raise TypeError('Wrong type for random_state')
    seed_list = rng.randint(1, np.iinfo(np.int32).max, no_bootstraps)
    return seed_list


def empirical_p_value(W, offset = 0):
    ''' 
    input: W (p, ): p is the number of hypothesis/variables
        Generating intermediate p-values from the statistics W
        if W_j >0: r_j = 1 + #{k: W_k <= - W_j} /p
        else W_j<=0: r_j = 1
    output: intermediate p-values, r
    '''
    p_vals = []
    n = W.size
    if offset not in (0, 1):
        raise ValueError("offset should be 0 or 1")
    W_inv = - W
    for i in range(n):
        if W[i] <= 0:
            p_vals.append(1)
        else:
            p_vals.append((offset + np.sum(W_inv >= W[i])) / n)
    return np.array(p_vals) 

def fdr_threshold(pvals, fdr = 0.1, method = 'bhq', reshaping_function = None):
    if method == 'bhq':
        return bhq_threshold(pvals, fdr = fdr)
    elif method == 'bhy':
        return bhy_threshold(
            pvals, fdr=fdr, reshaping_function=reshaping_function)
    else:
        raise ValueError(
            '{} is not support FDR control method'.format(method))

def bhq_threshold(pvals, fdr = 0.1):
    '''
    order pvals in ascending order r_1<r_2<...< r_p
    r'_k <= k * alpha / p
    tau = r'_k is the threshold
    '''
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if pvals_sorted[i] <= fdr * (i + 1) / n_features:
            selected_index = i
            break
    if selected_index <= n_features:
        return pvals_sorted[selected_index]
    else:
        return -1.0
    
def bhy_threshold(pvals, reshaping_function=None, fdr=0.1):
    '''
    order pvals in ascending order r_1<r_2<...< r_p
    r'_k <= k * alpha /{p * (sum(1 / i[i = 1 to i=p])}
    tau = r'_k is the threshold
    '''
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    if reshaping_function is None:
        temp = np.arange(n_features)
        sum_inverse = np.sum(1 / (temp + 1))
        return bhq_threshold(pvals, fdr / sum_inverse)
    else:
        for i in range(n_features - 1, -1, -1):
            if pvals_sorted[i] <= fdr * reshaping_function(i + 1) / n_features:
                selected_index = i
                break
    if selected_index <= n_features:
        return pvals_sorted[selected_index]
    else:
        return -1.0
    
def quantile_aggregation(p_vals, gamma = 0.5, gamma_min = 0.05, adaptive = False):
    '''
    p_vals : (no_bootstraps , n_test)
    gamma : percentile value for aggregation
    
    return: 1D ndarray (n_tests, ) Vector of aggregated p-value
    Ref: quantile aggregation method, Meinshausen et al. (2008)
    '''
    if adaptive:
        
        gammas = np.arange(gamma_min, 1.05, 0.05)
        list_Q = np.array([fixed_quantile_aggregation(p_vals, gamma) for gamma in gammas])
        return np.minimum(1, (1 - np.log(gamma_min)) * list_Q.min(0))
    
    else:
        return fixed_quantile_aggregation(p_vals, gamma)
        
def fixed_quantile_aggregation(p_vals, gamma):
    converted_score = (1 /gamma) * (np.percentile(p_vals, q = 100 * gamma, axis = 0))
    return np.minimum(1, converted_score)