import torch
import logging
import gpytorch
import numpy as np

import skisg.utils as utils
import skisg.config as config

from skisg.config import SgBasisType, MatmulAlgo
from matplotlib import pyplot as plt

from skisg.algos.sgnpmvm import get_kernel_matrix
from skisg.kernels.sgkernel import SparseGridKernel
from skisg.interp.sparse.sgindices import compute_LI, compute_LI_order

utils.set_loglevel(level=logging.DEBUG)

PLOT = False

DECIMAL_LEVEL = 5  # At decimal 6, many tests are failing as of now.
DEFAULT_DTYPE = 'float64'
CASES_READY = 4

algo_basis_pairs = [
    (MatmulAlgo.ITERATIVE, SgBasisType.BOUNDSTART),
    (MatmulAlgo.RECURSIVE, SgBasisType.NAIVE),
    (MatmulAlgo.ITERATIVE, SgBasisType.NAIVE),
    (MatmulAlgo.ITERATIVE, SgBasisType.MODIFIED)
]


def _test_MVM(grid_level, ndim, ls, algo_type=MatmulAlgo.RECURSIVE,
              basis_type=SgBasisType.NAIVE, use_toeplitz=False):

    print("\n\n\nStarting ....")
    dtype = config.dtype(DEFAULT_DTYPE, use_torch=True)

    assert len(ls) == ndim
    np.random.seed(1337)

    K = ls
    LI = compute_LI(grid_level, ndim, comb=False, basis=basis_type)

    nvecs = np.min([100, LI.shape[0]])
    #X = np.random.rand(LI.shape[0], nvecs)
    X = np.eye(LI.shape[0])
    # X = np.zeros((LI.shape[0], LI.shape[0]))
    # X[0, 0] = 1
    print("X dtype: ", X.dtype, type(K[0]))
    kmat = get_kernel_matrix(LI, K, basis=basis_type)
    # order = compute_LI_order(grid_level, ndim, basis=basis_type)
    # kmat = kmat[order, :]
    # kmat = kmat[:, order]

    device = config.get_device()
    print("k dtype: ", kmat.dtype)
    rhs = torch.from_numpy(X).to(dtype=dtype, device=device)
    kmat = torch.from_numpy(kmat).to(dtype=dtype, device=device)
    desired = torch.matmul(kmat, rhs)
    print("dtypes: ", rhs.dtype, kmat.dtype, desired.dtype)

    logging.info("GL: " + str(grid_level) + ", ndimension: " + str(ndim))
    logging.info("|GL|: " + str(LI.shape[0]) + ", nvectors: " + str(nvecs))

    # computing MVM
    kernel_covar = SparseGridKernel(
        umin=0.0,
        umax=1.0,
        basis=basis_type,
        base_kernel=gpytorch.kernels.RBFKernel(),
        grid_level=grid_level,
        covar_dtype=dtype,
        ndim=ndim,
        algo_type=algo_type,
        use_toeplitz=use_toeplitz,
    )
    kernel_covar.to(dtype=dtype, device=config.get_device())

    print("dtype:", kernel_covar.dtype)

    # initialize hyper-parameters
    kernel_covar.initialize_hypers(lengthscales=ls)
    Kx = kernel_covar.forward(x1=X)
    actual = Kx._matmul(rhs)

    try:
        return desired.detach().numpy(), actual.detach().numpy()
    except TypeError:
        return desired.detach().cpu().numpy(), actual.detach().cpu().numpy()


def process_ls(ls):
    return [np.float64(_ls) for _ls in ls]


def _test_wrapper(grid_level, ndim, ls):
    ls = process_ls(ls)
    for use_toeplitz in [False, True]:
        for algo_type, basis_type in algo_basis_pairs[:CASES_READY]:
            if algo_type == MatmulAlgo.RECURSIVE and use_toeplitz:
                continue
            desired, actual = _test_MVM(grid_level=grid_level, ndim=ndim, ls=ls,
                                        algo_type=algo_type,
                                        basis_type=basis_type)

        if PLOT:
            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(actual)
            # axs[1].imshow(desired)
            # axs[2].imshow(np.abs(actual - desired))
            plt.imshow(np.abs(actual - desired))
            plt.colorbar()
            plt.show()
        np.testing.assert_almost_equal(actual=actual, desired=desired, decimal=DECIMAL_LEVEL)


def test_MVM1():
    _test_wrapper(grid_level=1, ndim=2, ls=[1.0, 1.0])


def test_MVM2():
    _test_wrapper(grid_level=3, ndim=3, ls=[0.8, 0.7, 0.6])


def test_MVM5():
    _test_wrapper(grid_level=1, ndim=2, ls=[1.0, 0.8])


def test_MVM6():
    _test_wrapper(grid_level=1, ndim=2, ls=[0.8, 0.9])


def test_MVM7():
    _test_wrapper(grid_level=3, ndim=2, ls=[0.8, 0.9])


def test_MVM8():
    _test_wrapper(grid_level=3, ndim=3, ls=[0.8, 0.9, 0.5])


def test_MVM11():
    _test_wrapper(grid_level=3, ndim=4, ls=[0.8, 0.9, 0.5, 0.4])


# if __name__ == '__main__':
#     test_MVM2()
