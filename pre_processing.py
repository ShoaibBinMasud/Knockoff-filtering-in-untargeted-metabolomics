from sklearn.impute import KNNImputer
from scipy.stats import iqr

# missing value imputation using KNN
def missing_value_imputation(data):
    x = data.to_numpy()
    imputer = KNNImputer(n_neighbors = 2, weights="uniform")
    imputed_x =  imputer.fit_transform(x)
    new_data = data.copy()
    new_data.iloc[:, :] = imputed_x
    return new_data

# Removing metabolites based on Interquartile range filter
def remove_features_iqr(data):
    
    new_df2 = data
    
    if len(new_df2.columns) <= 500:
        no_of_columns_to_drop = 0
        print("No need to apply interquartile range filter")

    elif len(new_df2.columns)>= 1000 and len(new_df2.columns) <= 5000:
        no_of_columns_to_drop = int(len(new_df2.columns) * 15 /100)
        
    elif len(new_df2.columns)>= 5000 and len(new_df2.columns) <= 10000:
        no_of_columns_to_drop = int(len(new_df2.columns) * 30 /100)
        
    else:
        no_of_columns_to_drop = int(len(new_df2.columns) * 40 /100)
    '''
    x = np.array([[10,  7,  4],
                  [ 3,  2,  1]])
    iqr(x, axis=0)
    array([ 3.5,  2.5,  1.5])
    '''
    interquartile_range = iqr(new_df2.values, axis = 0)
    drop_columns = interquartile_range.argsort()[:no_of_columns_to_drop]
    cols = new_df2.columns[drop_columns]
    new_df2.drop(columns = cols,  inplace = True, axis=1)
    print('Removed %d metabolites with least interquartile range' %no_of_columns_to_drop)
    return new_df2