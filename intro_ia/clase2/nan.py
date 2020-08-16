import numpy as np

def adding_nan(matrix, n_samples, percentage):
    percentage_float = percentage/100
    n_NaN_samples = int(n_samples * percentage_float)
    row = np.random.randint(n_samples, size=n_NaN_samples)
    column = np.random.randint(4, size=n_NaN_samples)
    matrix[row, column] = np.nan
    return matrix
    
def nan_search(matrix):     
    nan_data = np.isnan(matrix)
    amount_nan_data = np.sum(np.sum(nan_data, axis=1),axis=0)
    print('Cantidad de NaN data = ', format(amount_nan_data))

def replace_nan_with_mean(matrix, n_samples, feactures):
    mean_filtered_data = np.zeros(feactures)
    nan_row,nan_col = np.where(np.isnan(matrix))
    unique_nan_col = np.unique(nan_col)
    idx_col_nan = np.bincount(unique_nan_col)
    zero_extended = np.zeros(len(mean_filtered_data)-len(idx_col_nan))
    idx_col_nan_extended = np.concatenate((idx_col_nan,zero_extended), axis=0)
    col_nan_mask = idx_col_nan_extended == 0
    total_col = np.arange(feactures)
    data_valid_col = total_col[col_nan_mask] 
    mean_filtered_data[data_valid_col] = np.mean(matrix[:,data_valid_col], axis=0)
    
    nan_col_amount = len(unique_nan_col)
    total_row = np.arange(n_samples)
    idx_row_nan = np.zeros((n_samples,nan_col_amount))
    
    idx_row_nan = np.bincount(nan_row,unique_nan_col)
    data_valid_row_mask = ~np.isnan(matrix[:,unique_nan_col])
    nan_row_mask = np.isnan(matrix[:,unique_nan_col])
    
    for i in range(unique_nan_col.shape[0]):
        mean_filtered_data[unique_nan_col[i]] = np.mean(matrix[data_valid_row_mask[:,i],unique_nan_col[i]], axis=0)
        matrix[nan_row_mask[:,i], unique_nan_col[i]] = mean_filtered_data[unique_nan_col[i]]
    
def delete_nan(matrix):
    nan_row,nan_col = np.where(np.isnan(matrix))
    filtered_data = np.delete((np.delete(matrix, nan_row, axis=0)), nan_col, axis = 1)
    matrix = filtered_data  