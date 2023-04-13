import logging
import numpy as np

from skisg.utils import set_loglevel, log_np

from skisg.interp.sparse.construct import G
from skisg.interp.sparse.sgindices import compute_LI
from skisg.algos.sgnpmvm import kernel_matrix_1d, get_kernel_matrix, fast_mvm_algorithm

set_loglevel(level=logging.DEBUG)


def _test_MVM(grid_level, ndim, ls):
    assert len(ls) == ndim

    np.random.seed(1337)

    nvecs = G(grid_level, ndim)
    K = ls
    LI = compute_LI(grid_level, ndim)

    nvecs = np.min([100, nvecs])
    X = np.random.rand(LI.shape[0], nvecs)

    logging.info("GL: " + str(grid_level) + ", ndimension: " + str(ndim))
    logging.info("|GL|: " + str(LI.shape[0]) + ", nvectors: " + str(nvecs))

    # computing MVM
    KX = fast_mvm_algorithm(K=K, X=X, LI=LI, gl=grid_level, dim=ndim)
    kmat = get_kernel_matrix(LI, K)

    print("Kmat: ", kmat.dtype)
    print("X: ", X.dtype)
    print("KX: ", KX.dtype)
    print("MVM:", np.matmul(kmat, X).dtype)
    # print("error: ", np.abs(KX - np.matmul(kmat, X)))

    return np.matmul(kmat, X), KX


def test_kernel_matrix_1d():
    grid_level = 3
    ls = 1.0
    LI = compute_LI(grid_level, ndim=1)
    actual = kernel_matrix_1d(gl=grid_level, ls=ls)
    desired = get_kernel_matrix(LI, [ls])
    np.testing.assert_almost_equal(actual, desired)


def test_MVM():

    desired, actual = _test_MVM(grid_level=4, ndim=1, ls=[1])
    np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=4, ndim=1, ls=[0.8])
    np.testing.assert_almost_equal(actual=actual, desired=desired)


def test_MVM2D():
    # test for 2-D kernel matrix construction
    # logging.info("Kernel matrix test 2-D: ")
    # desired, actual = _test_MVM(grid_level=1, ndim=2, ls=[1.0, 1.0])
    # np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=1, ndim=2, ls=[0.8, 0.9])
    np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=2, ndim=2, ls=[0.8, 0.9])
    np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=3, ndim=2, ls=[0.8, 0.9])
    np.testing.assert_almost_equal(actual=actual, desired=desired)


def test_MVM3D():
    logging.info("Kernel matrix test 3-D: ")
    desired, actual = _test_MVM(grid_level=1, ndim=3, ls=[1.0, 1.0, 1.0])
    np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=2, ndim=3, ls=[1.0, 1.0, 1.0])
    np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=2, ndim=3, ls=[0.9, 0.8, 0.7])
    np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=3, ndim=3, ls=[0.4, 0.8, 0.7])
    np.testing.assert_almost_equal(actual=actual, desired=desired)


def test_MVM4D():
    logging.info("Kernel matrix test 4-D: ")
    desired, actual = _test_MVM(grid_level=1, ndim=4, ls=[1.0, 1.0, 1.0, 1.0])
    np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=3, ndim=4, ls=[1.0, 1.0, 1.0, 1.0])
    np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=4, ndim=4, ls=[0.9, 0.8, 0.7, 0.6])
    np.testing.assert_almost_equal(actual=actual, desired=desired)

    desired, actual = _test_MVM(grid_level=5, ndim=4, ls=[0.4, 0.8, 0.7, 0.5])
    np.testing.assert_almost_equal(actual=actual, desired=desired)
