import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression


def rf_oob_score(X, X_knockoff, y):
    
    p = X.shape[1]
    clf = RandomForestClassifier(n_estimators = 1000, bootstrap = True, oob_score = True, max_features = 'sqrt')
    clf.fit(np.hstack((X, X_knockoff)), y)
    Z = clf.feature_importances_
    W = np.abs(Z[:p]) - np.abs(Z[p : (2 * p)])
    return Z, W

def plsda_score(X, X_knockoff, y, n_components = 2, stat = 'vip'):
    
    p = X.shape[1]
    unique_label =  np.unique(y)
    new_y = np.zeros((len(y), len(unique_label)))
    for label in unique_label:
        i =  np.where(y == label)[0]
        new_y[i, int(label)] = 1  
        
    pls = PLSRegression(n_components)
    pls.fit(np.hstack((X, X_knockoff)), new_y)
    
    weights = pls.x_weights_
    x_scores = pls.x_scores_
    y_loadings = pls.y_loadings_
    x_loadings = pls.x_loadings_
    
    W0 = weights / np.sqrt(np.sum(weights ** 2, axis=0))
    sumSq = np.sum(x_scores ** 2, axis=0) * np.sum(y_loadings ** 2, axis=0)
    vip = np.sqrt(len(x_loadings) * np.sum(sumSq * W0 ** 2, axis=1) / np.sum(sumSq, axis=0))
    
    
    
    if stat == 'vip':
        return vip, np.abs(vip[:p]) - np.abs(vip[p : 2 * p])
    else:
        return x_loadings, np.abs(x_loadings[:p]) - np.abs(x_loadings[p : 2 * p])