# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 22:01:55 2020

@author: crist
"""

import numpy as np
import matplotlib.pyplot as plt

inv_overlap = 2
n_samples = 100
centroids = np.array([[1,0,0,0],[0,1,0,0],], dtype=np.float32)
centroids = centroids * inv_overlap
data = np.repeat(centroids, n_samples / 2, axis=0)
normal_noise = np.random.normal(loc=0, scale=1, size=(n_samples, 4))
data = data + normal_noise
cluster_ids = np.array([[0],[1]])
cluster_ids = np.repeat(cluster_ids, n_samples / 2, axis=0)


fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(data[:, 0], data[:, 1], c=cluster_ids, s=50)
fig.show()

#%%--------------------------------------------------------------------------
#Ejercicio 5

size = 100
def exponential_random_variable(lambda_param, size):
    uniform_random_variable = np.random.uniform(low=0.0, high=1.0, size=size)
return (-1 / lambda_param) * np.log(1 - uniform_random_variable)

#plot
lambda_param = 2
x_exp=(-1 / lambda_param) * np.log(1 - uniform_random_variable)
samples = np.arange(x_exp.shape[0])
plt.plot(samples, (-1 / lambda_param) * np.log(1 - uniform_random_variable), linewidth=2, color='m')
plt.show()

#%%--------------------------------------------------------------------------
#Ejercicio 6

def exponential_random_variable(lambda_param, size):
    uniform_random_variable = np.random.uniform(low=0.0, high=1.0, size=size)
return (-1 / lambda_param) * np.log(1 - uniform_random_variable**3)

#%%--------------------------------------------------------------------------
#Ejercicio 7

size = 100
def exponential_random_variable(random_variable, size):
    mean = np.mean(random_variable, axis=0)
    std = np.std(random_variable, axis=0)
return (random_variable - mean)/std

                                            
#%%-------------------------------------------------------------------------

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='m')
plt.show()
