"""
Doctests for hw0_knn.py
No changes needed to this file.

# Define example with N=4 and Q=2
>>> data_NF = np.asarray([
...     [ 1.,  0.],
...     [ 0.,  1.],
...     [-1.,  0.],
...     [ 0., -1.]])
>>> query_QF = np.asarray([
...     [0.9,  0.0],
...     [0.0, -0.9]])

Example Test K=1
----------------
# Find the single nearest neighbor for each query vector
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=1)
>>> neighb_QKF.shape
(2, 1, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[1., 0.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.]])

Example Test K=3
----------------
# Now find 3 nearest neighbors for the same queries
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=3)
>>> neighb_QKF.shape
(2, 3, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[ 1.,  0.],
       [ 0.,  1.],
       [ 0., -1.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.],
       [ 1.,  0.],
       [-1.,  0.]])

"""

import numpy as np
from hw0_knn import calc_k_nearest_neighbors
