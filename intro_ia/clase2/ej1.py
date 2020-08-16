import numpy as np

X = np.array([
[1,2,3],
[4,5,6],
[7,8,9]
])
C = np.array([
[1,0,0],
[0,1,1]
])
expanded_C = C[:, None]
distances = np.sqrt(np.sum((expanded_C - X) ** 2, axis=2))
print(distances)
# [[ 3.60555128 8.36660027 13.45362405]
# [ 2.44948974 7.54983444 12.72792206]]


arg_min = np.argmin(distances, axis=1)
print(arg_min)
