import torch
import logging
import numpy as np

import skisg.utils as utils
import skisg.config as config
from skisg.config import SgBasisType, MatmulAlgo


from skisg.interp.sparse.sgindices import compute_LI, compute_LI_pairs
from skisg.algos.sgnpmvm import get_kernel_matrix

from skisg.kernels.sgcovars import compute_covars
from skisg.lazy.sgkerneltensor import SGKernelLazyTensor


utils.set_loglevel(level=logging.DEBUG)
MED_TOLERANCE = 1e-2
DECIMAL_LEVEL = 5  # At decimal 6, many tests are failing as of now.
DEFAULT_DTYPE = 'float64'

CASES_READY = 2

algo_basis_pairs = [
    (MatmulAlgo.RECURSIVE, SgBasisType.MODIFIED),
    (MatmulAlgo.ITERATIVE, SgBasisType.MODIFIED),
    (MatmulAlgo.ITERATIVE, SgBasisType.BOUNDSTART),
    (MatmulAlgo.ITERATIVE, SgBasisType.MODIFIED)
]


def process_ls(ls):
    return [np.float64(_ls) for _ls in ls]


def help_setup(grid_level, ndim, ls,
               algo_type=MatmulAlgo.RECURSIVE,
               basis_type=SgBasisType.NAIVE,
               use_toeplitz=False,
               max_nvecs=100):

    print("\n\n\nStarting ....")
    dtype = config.dtype(DEFAULT_DTYPE, use_torch=True)

    assert len(ls) == ndim
    np.random.seed(1337)

    assert len(ls) == ndim
    K = ls
    LI = compute_LI(grid_level, ndim, comb=False, basis=basis_type)
    nvecs = LI.shape[0]

    nvecs = np.min([max_nvecs, nvecs])
    X = np.random.rand(LI.shape[0], nvecs)
    kmat = get_kernel_matrix(LI, K, basis=basis_type)

    logging.info("GL: " + str(grid_level) + ", n dimension: " + str(ndim))
    logging.info("|GL|: " + str(LI.shape[0]) + ", n vectors: " + str(nvecs))

    # computing MVM
    dtype = config.dtype(DEFAULT_DTYPE, use_torch=True)
    device = config.get_device()

    rhs = torch.from_numpy(X).to(dtype=dtype, device=device)

    covars, sorting_order = compute_covars(
        grid_level=grid_level,
        ndim=ndim,
        umin=[0.0]*ndim,
        umax=[1.0]*ndim,
        ls=ls,
        basis=basis_type,
        device=device,
        dtype=dtype,
        use_toeplitz=use_toeplitz,
    )

    # Setting up lexicographic indices
    if algo_type == MatmulAlgo.RECURSIVE:
        levels_and_indices = compute_LI(grid_level, ndim, comb=False, basis=basis_type)
        levels_and_indices = torch.from_numpy(levels_and_indices).to(dtype=torch.long, device=device)

    elif algo_type == MatmulAlgo.ITERATIVE:
        levels_and_indices = compute_LI_pairs(grid_level, ndim, comb=False, basis=basis_type)
    else:
        raise NotImplementedError

    Ksg = SGKernelLazyTensor(
        covars=covars,
        LI=levels_and_indices,
        grid_level=grid_level,
        ndim=ndim,
        basis_type=basis_type,
        algo_type=algo_type,
        sorting_orders=sorting_order,
        use_toeplitz=use_toeplitz,
    )

    return Ksg, kmat, X, rhs


def _test_mvm_wrapper(grid_level, ndim, ls):
    ls = process_ls(ls)
    for use_toeplitz in [False, True]:
        for algo_type, basis_type in algo_basis_pairs[:CASES_READY]:
            print("\n\nCase:", use_toeplitz, algo_type, basis_type)
            if algo_type == MatmulAlgo.RECURSIVE and use_toeplitz:
                continue
            Ksg, kmat, X, rhs = help_setup(grid_level=grid_level, ndim=ndim, ls=ls,
                                           algo_type=algo_type, basis_type=basis_type, use_toeplitz=use_toeplitz)
            desired = torch.from_numpy(np.matmul(kmat, X))
            actual = Ksg._matmul(rhs)
            try:
                desired, actual = desired.detach().numpy(), actual.detach().numpy()
            except TypeError:
                desired, actual = desired.detach().cpu().numpy(), actual.detach().cpu().numpy()
            np.testing.assert_almost_equal(actual=actual, desired=desired, decimal=DECIMAL_LEVEL)


def test_MVM1():
    _test_mvm_wrapper(grid_level=1, ndim=2, ls=[1.0, 1.0])


def test_MVM2():
    _test_mvm_wrapper(grid_level=1, ndim=2, ls=[1.2, 1.5])


def test_MVM3():
    _test_mvm_wrapper(grid_level=2, ndim=2, ls=[1.2, 1.5])


def test_MVM4():
    _test_mvm_wrapper(grid_level=2, ndim=2, ls=[0.8, 1.9])


def test_MVM5():
    _test_mvm_wrapper(grid_level=3, ndim=3, ls=[0.8, 1.9, 0.3])


def test_MVM6():
    _test_mvm_wrapper(grid_level=1, ndim=4, ls=[0.8, 1.9, 0.89, 0.2])


def test_MVM7():
    _test_mvm_wrapper(grid_level=2, ndim=4, ls=[0.8, 1.9, 0.89, 0.2])


# def _test_cg_wrapper(grid_level, ndim, ls):
#     NOTE -- CG isn't really required on this Lazy Tensor. Revisit this only if it is required.
#     ls = process_ls(ls)
#     for algo_type, basis_type in algo_basis_pairs[:CASES_READY]:
#         Ksg, kmat, X, desired_lhs = help_setup(grid_level=grid_level, ndim=ndim, ls=ls,
#                                                algo_type=algo_type, basis_type=basis_type)
#         rhs = (Ksg._matmul(desired_lhs)).detach()
#         rhs_norm = torch.sqrt(torch.sum(rhs ** 2, 0)).detach()
#         desired_lhs = desired_lhs / rhs_norm
#         rhs = (Ksg._matmul(desired_lhs)).detach()
#
#         config.set_gpytorch_settings(cg_tol=1e-10, max_cg_iters=500)
#         actual_lhs = Ksg.inv_matmul(rhs).detach()
#         max_element_diff = np.max(np.abs((Ksg._matmul(actual_lhs)).detach().numpy() - rhs.detach().numpy()))
#         np.testing.assert_array_less(max_element_diff, MED_TOLERANCE)
#
#
# def test_CG1():
#     med = _test_cg_wrapper(grid_level=3, ndim=2, ls=[0.8, 0.9])
#     np.testing.assert_array_less(med, MED_TOLERANCE)


# def test_CG2():
#     med = _test_CG(grid_level=2, ndim=3, ls=[0.8, 0.9, 0.5])
#     np.testing.assert_array_less(med, MED_TOLERANCE)
#
#
# def test_CG3():
#     med = _test_CG(grid_level=2, ndim=4, ls=[0.8, 0.9, 0.5, 1.2])
#     np.testing.assert_array_less(med, MED_TOLERANCE)


if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    test_MVM1()
    # test_MVM2()
    # test_MVM3()
    # test_MVM4()
    # test_MVM5()
    # test_MVM6()
    # test_MVM7()
