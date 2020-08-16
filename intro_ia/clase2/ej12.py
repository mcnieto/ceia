import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from synthetic_dataset import SyntheticDataset
from random_variable import exponential_random_variable
from k_means import k_means
from norm import vector_norm_l0, vector_norm_l1, vector_norm_l2, vector_norm_inf
from nan import adding_nan

import pickle

if __name__ == '__main__':
    # create the synthethic dataset (dimensions = 4, samples = 100K)
    synthetic_dataset = SyntheticDataset(n_samples=10, inv_overlap=10)   
    print('Dataset Creado, n_samples = ', format(synthetic_dataset.n_samples))
    #nan_search(synthetic_dataset.data)
    
    # adding NaN (0.1%)
    nan_percentage=20
    adding_nan(synthetic_dataset.data, synthetic_dataset.n_samples, nan_percentage)
    #nan_search(synthetic_dataset.data)

    #save dataset in file
    synthetic_dataset.save_file()
    
    #load dataset from file
    with open('dataset.pkl', 'rb') as file:
        dataset_from_file = pickle.load(file)
    
    #reemplace NaN with reans values
    replace_nan_with_mean(synthetic_dataset.data, synthetic_dataset.n_samples, feactures=4)
    
    #cargo solo los datos de entrenamiento (70%)
    train, train_cluster_ids, valid, valid_cluster_ids = synthetic_dataset.train_valid_split()
    
    #norma l2 by feacture
    train_norm_l2 = vector_norm_l2(train)
    
    #agrego columna con vectores aleatorios de todos los puntos
    exponential_rv = exponential_random_variable(lambda_param=1, size=train.shape[0])
    train_expanded = np.concatenate((train, exponential_rv.reshape(-1,1)), axis=1)
    
    #histogram
    fig = plt.figure()
    plt.hist(exponential_rv, bins='auto')
    fig.savefig("exponential_rv_histogram.png")
    
    #%%
    # apply pca over the extended dataset to plot the cluster
    train_expanded_pca = SyntheticDataset.reduce_dimension(train_expanded, 2)
    SyntheticDataset.plot_cluster(train_expanded_pca, train_cluster_ids)
