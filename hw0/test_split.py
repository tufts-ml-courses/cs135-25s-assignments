"""
Doctests for hw0_split.py
No changes needed to this file.

Example Test L=6
-----------------
# L := total number of data instances
# F := dimension of each feature vector
>>> x_LF = np.asarray([
... [ 0, 11.],
... [ 0, 22.],
... [ 0, 33.],
... [-2, 44.],
... [-2, 55.],
... [-2, 66.],
... ])
>>> xcopy_LF = x_LF.copy() # preserve what input was before the call
>>> train_MF, test_NF = split_into_train_and_test(
...     x_LF, frac_test=2/6, random_state=np.random.RandomState(0))
>>> train_MF.shape
(4, 2)
>>> test_NF.shape
(2, 2)
>>> print(train_MF)
[[-2. 66.]
 [ 0. 33.]
 [ 0. 22.]
 [-2. 44.]]
>>> print(test_NF)
[[ 0. 11.]
 [-2. 55.]]

# Verify that input array did not change due to function call
>>> np.allclose(x_LF, xcopy_LF)
True

"""

import numpy as np
from hw0_split import split_into_train_and_test
