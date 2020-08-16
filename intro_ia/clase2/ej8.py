import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

x = np.array([ [0.4, 4800, 5.5], [0.7, 12104, 5.2], [1, 12500, 5.5], [1.5, 7002, 4.0] ])

pca = PCA(n_components=2)
x_std = StandardScaler(with_std=False).fit_transform(x)
pca_out = pca.fit_transform(x_std)

x2 = (x - x.mean(axis=0))
cov_1 = np.cov(x2.T)
w, v = np.linalg.eig(cov_1)
idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]
x_low = np.matmul(x2, v[:, :2])

print(pca_out)

print(x_low)
