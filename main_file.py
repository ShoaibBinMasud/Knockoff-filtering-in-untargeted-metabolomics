import numpy as np
import pandas as pd
import warnings
import glob
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from utils_aggregate import (generate_seedlist, empirical_p_value, quantile_aggregation, fdr_threshold)
from create_gaussian_knockoff import(estimating_distribution,gaussian_knockoff_generation )
from knockoff_test_statistics import rf_oob_score
from pre_processing import(missing_value_imputation, remove_features_iqr)
warnings.filterwarnings("ignore")

myFiles = glob.glob('1st_Round_Research/*.txt')
mydict = dict()
writer = pd.ExcelWriter('2nd_Round_Research_metabolites_list.xlsx', engine='xlsxwriter')
for f in myFiles:
    dataset_name = f.split('\\')[1].split('.')[0]
    print(dataset_name)
    df =  pd.read_csv(f, sep='\t', dtype = object)
    df = df.rename(columns = df.iloc[0])
    df = df.drop(df.index[0]).reset_index(drop = True)
    df.columns = df.columns.str.replace(' ', '').str.lower()
    diagnose = df.columns[1:].str.split('|').str[0]
    diagnose= diagnose.str.split(':').str[1]
    _, label = np.unique(diagnose, return_inverse=True)
    df = df.set_index('factors')
    print('unique diagnosis', diagnose.unique(), '\tMetabolite X Samples: ' , df.shape)
    
    # Remove those metabolites which atleast are filled with 70% of NAN values
    #lambd = [0, 0.6, 0.7, 0.8, 1]
    lambd = [0, 1]
    meta_dict = dict()
    for k in lambd:
        data =  df.copy().T
        no_of_samples =  data.shape[0]
        thresh = int(no_of_samples * k)
        data = data.dropna(axis = 1, thresh = thresh)#keeping the metabolites which has atleast k percenet filled values
        missing_percentage_after = data.isnull().sum().sum()/ (data.shape[0] * data.shape[1])
        print('After: (Metabolite X Samples): ' , data.shape, '\t\t percentage of missing values: %.3f'
              %missing_percentage_after,'\n')
        
        # missing value imputation based on K-nearest neighbours
        imputed_data = missing_value_imputation(data)
        print('After: (Metabolite X Samples): ' , data.shape)
        
        # apply interquartile range filter 
        filtered_data = remove_features_iqr(imputed_data)
        print('After: (Samples X Metabolites): ' , filtered_data.shape)
        
        # standardize the feature
        standard_data = filtered_data.copy()
        standard_data.iloc[:, :] = StandardScaler().fit_transform(standard_data)
        
        # estimating the distribution
        X = standard_data.values
        mu, Sigma = estimating_distribution(X, shrink = False, cov_estimator = 'ledoit_wolf')
        no_bootstraps = 10  
        no_jobs = 2 
        seed_list = generate_seedlist(no_bootstraps, random_state = None)
        
        X_knockoff = Parallel(n_jobs = no_jobs, prefer="threads")(delayed(gaussian_knockoff_generation)
                                                                  (X, mu, Sigma, method = 'equi') for seed in seed_list)
        
        r =Parallel(n_jobs=no_jobs, prefer="threads")(delayed(rf_oob_score)(X, X_knockoff[i], label) 
                                                      for i in range(no_bootstraps))
        Z, W = zip(*r) 
        p_vals = np.array([empirical_p_value(W[i], offset = 0) for i in range(no_bootstraps)])
        aggregated_pvals = quantile_aggregation(p_vals, gamma = 0.4, gamma_min = 0.05, adaptive = True)
        threshold = fdr_threshold(aggregated_pvals, fdr = 0.1, method = 'bhq')
        S = np.where(aggregated_pvals <= threshold)[0]
        meta_list_aggreagate = list(standard_data.columns[S]) 
        print(len(meta_list_aggreagate))
        meta_dict[(k, X.shape[1])]= meta_list_aggreagate
    meta_df = pd.DataFrame.from_dict(meta_dict, orient='index')
    meta_df.T.to_excel(writer, sheet_name = dataset_name )
writer.save()  