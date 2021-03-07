import numpy as np
import cvxpy as cvx
import warnings
import scipy.linalg as la
from sklearn.covariance import (GraphicalLassoCV, empirical_covariance, ledoit_wolf)
#from statsmodels.stats.moment_helpers import cov2corr


def estimating_distribution(X, shrink = False, cov_estimator='ledoit_wolf'):
    alphas = [1e-3, 1e-2, 1e-1, 1]
    mu = X.mean(axis = 0)
    Sigma = empirical_covariance(X)

    if shrink or not check_posdef(Sigma):
        if cov_estimator == 'ledoit_wolf':
            print('ledoit_wolf')
            Sigma_shrink = ledoit_wolf(X, assume_centered=True)[0]
        elif cov_estimator == 'graph_lasso':
            print('1')
            model = GraphicalLassoCV(alphas=alphas)
            Sigma_shrink = model.fit(X).covariance_
        else:
            raise ValueError('{} is not a valid covariance estimated method'.format(cov_estimator))
        return mu, Sigma_shrink

    return mu, Sigma

def gaussian_knockoff_generation(X, mu, Sigma, method='equi'):
    
    n, p = X.shape
    if method == 'equi':
        diag_s = np.diag(create_equi_s(Sigma))
    elif method == 'sdp':
        diag_s = np.diag(create_sdp_s(Sigma))
        
    SigmaInv_diag = la.lstsq(Sigma, diag_s)[0]
    sigma_tilde = 2.0 * diag_s - np.dot(diag_s, SigmaInv_diag)
    LV = np.linalg.cholesky(sigma_tilde + 1e-10 * np.eye(p))
    while not check_posdef(sigma_tilde):
        sigma_tilde += 1e-10 * np.eye(p)
        warnings.warn('Adding minor positive value to the matrix to make sigma_tilde positive definite')
    
    muTilde = X - np.dot(X-np.tile(mu,(n,1)), SigmaInv_diag)
    N = np.random.normal(size = muTilde.shape)
    return muTilde + np.dot(N, LV.T)

def create_equi_s(Sigma):
    n_features = Sigma.shape[0]
    G = cov_to_corr(Sigma)
    eig_value = np.linalg.eigvalsh(G)
    lambda_min = np.min(eig_value[0])
    S = np.ones(n_features) * min(2 * lambda_min, 1)
    psd = False
    s_eps = 0
    while psd is False:
        psd = check_posdef(2 * G - np.diag(S * (1 - s_eps)))
        if not psd:
            if s_eps == 0:
                s_eps = 1e-08
            else:
                s_eps *= 10
    S = S * (1 - s_eps)
    return S * np.diag(Sigma)

def create_sdp_s(Sigma, tol = 1e-3):
    
    if(np.min(np.linalg.eigvals(Sigma))<0):
        corrMatrix = cov_to_corr(Sigma + (1e-8)*np.eye(Sigma.shape[0]))
    else:
        corrMatrix = cov_to_corr(Sigma)
        
    p,_ = corrMatrix.shape
    s = cvx.Variable(p)
    objective = cvx.Maximize(sum(s))
    constraints = [ 2.0*corrMatrix >> cvx.diag(s) + cvx.diag([tol]*p), 0<=s, s<=1]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='CVXOPT')
    
    assert prob.status == cvx.OPTIMAL
    s = np.clip(np.asarray(s.value).flatten(), 0, 1)
    return np.multiply(s, np.diag(Sigma))

def check_posdef(X, tol=1e-14):
    eig_value = np.linalg.eigvalsh(X)
    return np.all(eig_value > tol)

def cov_to_corr(Sigma):
 
    features_std = np.sqrt(np.diag(Sigma))
    Scale = np.outer(features_std, features_std)

    Corr_matrix = Sigma / Scale

    return Corr_matrix
