import numpy as np
from norm import vector_norm_l2
    def sorting_vectors_by_norm_l2(matrix):
        norm_l2 = vector_norm_l2(matrix)
        arg_sort = np.argsort(norm_l2)