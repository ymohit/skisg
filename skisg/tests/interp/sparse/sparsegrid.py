import numpy as np

from skisg.interp.sparse.sgindices import compute_LI
from skisg.algos.sgops import mat_X_i


def helptest_recursive_nd(grid_level=3, ndim=3):
    LI = compute_LI(grid_level, ndim)
    nvecs = 10
    X = np.random.rand(LI.shape[0], nvecs)
    Y = np.zeros_like(X)
    for gl in range(grid_level + 1):
        X_i, LI_minus_1 = mat_X_i(X, gl, LI)

        # testing LI_minus_1 with the parent structure
        assert np.sum(np.abs(LI_minus_1 - compute_LI(grid_level - gl, ndim - 1))) == 0
        Y[LI[:, 0] == gl] = X_i.reshape(-1, nvecs)
    assert np.sum(np.abs(X - Y)) == 0


def test_recursive_indexing():
    helptest_recursive_nd(grid_level=2, ndim=4)
    helptest_recursive_nd(grid_level=2, ndim=3)
    helptest_recursive_nd(grid_level=1, ndim=3)
    helptest_recursive_nd(grid_level=0, ndim=3)
    helptest_recursive_nd(grid_level=0, ndim=4)
    helptest_recursive_nd(grid_level=2, ndim=2)
    helptest_recursive_nd(grid_level=0, ndim=2)
    helptest_recursive_nd(grid_level=1, ndim=2)
