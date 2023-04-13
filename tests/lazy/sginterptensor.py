import torch
import logging
import numpy as np

import skisg.config as config
import skisg.utils as utils

from skisg.config import SgBasisType, MatmulAlgo, InterpType

from skisg.interp.sparse.sgindices import compute_LI, compute_LI_pairs
from skisg.algos.sgnpmvm import get_kernel_matrix

from skisg.interp.sparse.weights import compute_W
from skisg.kernels.sgcovars import compute_covars


from skisg.lazy.sgkerneltensor import SGKernelLazyTensor
from skisg.lazy.sginterptensor import SymSGInterpolatedLazyTensor, ASymSGInterpolatedLazyTensor


utils.set_loglevel(level=logging.DEBUG)
MED_TOLERANCE = 1e-2
DECIMAL_LEVEL = 5

DEFAULT_DTYPE = 'float64'


def process_ls(ls):
    return [np.float64(_ls) for _ls in ls]


def help_setup(grid_level, ndim, ls,
               sym=True, use_toeplitz=False,
               comb=True,
               basis_type=SgBasisType.MODIFIED,
               algo_type=MatmulAlgo.ITERATIVE,
               interp_type=InterpType.LINEAR):

    ls = process_ls(ls)
    assert len(ls) == ndim
    dtype = config.dtype(DEFAULT_DTYPE, use_torch=True)

    n_train = 128
    assert len(ls) == ndim
    K = ls
    LI = compute_LI(grid_level, ndim, comb=False, basis=basis_type)
    nvecs = LI.shape[0]

    nvecs = np.min([100, nvecs])
    X = np.random.rand(n_train, nvecs)
    kmat = get_kernel_matrix(LI, K, basis=basis_type, comb=False)

    logging.info("GL: " + str(grid_level) + ", ndimension: " + str(ndim))
    logging.info("|GL|: " + str(LI.shape[0]) + ", nvectors: " + str(nvecs))

    dtype = config.dtype(DEFAULT_DTYPE, use_torch=True)
    device = config.get_device()
    rhs = torch.from_numpy(X).to(dtype=dtype)

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
    if algo_type == MatmulAlgo.ITERATIVE:
        levels_and_indices = compute_LI_pairs(grid_level, ndim, comb=False, basis=basis_type)
    else:
        raise NotImplementedError

    if comb:
        if grid_level - ndim >= 0:
            select = levels_and_indices[grid_level - ndim, ndim]
            kmat = kmat[~select, :][:, ~select]

    base_lazy_tensor = SGKernelLazyTensor(
        covars=covars,
        LI=levels_and_indices,
        grid_level=grid_level,
        ndim=ndim,
        basis_type=basis_type,
        algo_type=algo_type,
        sorting_orders=sorting_order,
        use_toeplitz=use_toeplitz,
        comb=comb,
    )

    X_train = torch.from_numpy(np.random.rand(n_train, ndim)).to(dtype=dtype)
    W = compute_W(X_train, grid_level, ndim, comb=comb, basis=basis_type, interp_type=interp_type)

    # W = SparseInterpolation().sparse_interpolate(
    #         grid_level=grid_level,
    #         ndim=ndim,
    #         umin=0.0, umax=1.0,
    #         x_target=X_train,
    #         comb=comb,
    #         interp_type=interp_type,
    #         basis=SgBasisType.MODIFIED,
    # )

    if sym:
        Kinterp = SymSGInterpolatedLazyTensor(
            base_lazy_tensor=base_lazy_tensor,
            left_interp_coefficient=W
        )
        return W, base_lazy_tensor, rhs, kmat, Kinterp

    X_right = torch.from_numpy(np.random.rand(n_train, ndim)).to(dtype=dtype)
    W_right = compute_W(X_right, grid_level, ndim, is_left=False, comb=comb,
                        basis=basis_type, interp_type=interp_type)

    Kinterp = ASymSGInterpolatedLazyTensor(
        base_lazy_tensor=base_lazy_tensor,
        left_interp_coefficient=W,
        right_interp_coefficient=W_right
    )

    return (W, W_right), base_lazy_tensor, rhs, kmat, Kinterp


def _test_MVM(grid_level, ndim, ls, sym=True):

    for basis_type in [SgBasisType.MODIFIED]:
        for interp_type in [InterpType.LINEAR, InterpType.CUBIC, InterpType.SIMPLEX]:
            for comb_case in [True]:

                if not comb_case:
                    if interp_type != InterpType.LINEAR:
                        continue
                    if basis_type != SgBasisType.NAIVE:
                        continue

                print(basis_type, interp_type, comb_case)

                if basis_type == SgBasisType.BOUNDSTART:
                    if grid_level > 2:
                        grid_level = 2
                        print("gl reduced to ", grid_level, 'from ', grid_level+2)
                                        
                W, base_lazy_tensor, rhs, kmat, Kinterp \
                    = help_setup(grid_level, ndim, ls,
                                 sym=sym, basis_type=basis_type, interp_type=interp_type, comb=comb_case)

                actual = Kinterp._matmul(rhs)

                # Computing desired
                if sym:
                    W_left, W_right_X = W, W._t_matmul(rhs)
                else:
                    W_left, W_right = W
                    W_right_X = W_right._matmul(rhs)

                KWTX = torch.matmul(torch.from_numpy(kmat).to(dtype=config.dtype(use_torch=True)), W_right_X)
                desired = W_left.matmul(KWTX)

                try:
                    desired, actual = desired.detach().numpy(), actual.detach().numpy()
                except TypeError:
                    desired, actual = desired.detach().cpu().numpy(), actual.detach().cpu().numpy()

                np.testing.assert_almost_equal(actual=actual, desired=desired, decimal=DECIMAL_LEVEL)


def test_MVM1():

    _test_MVM(grid_level=2, ndim=2, ls=[0.8, 0.9])
    _test_MVM(grid_level=2, ndim=2, ls=[0.8, 0.9], sym=False)


def test_MVM2():

    _test_MVM(grid_level=3, ndim=2, ls=[0.8, 0.9])
    _test_MVM(grid_level=3, ndim=2, ls=[0.8, 0.9], sym=False)


def test_MVM3():

    _test_MVM(grid_level=2, ndim=3, ls=[0.8, 0.9, 0.5])
    _test_MVM(grid_level=2, ndim=3, ls=[0.8, 0.9, 0.5], sym=False)


def test_MVM4():

    _test_MVM(grid_level=2, ndim=3, ls=[0.8, 0.5, 0.7])
    _test_MVM(grid_level=2, ndim=3, ls=[0.8, 0.9, 0.7], sym=False)


def test_MVM5():

    _test_MVM(grid_level=5, ndim=4, ls=[0.8, 0.9, 0.7, 0.2])
    _test_MVM(grid_level=5, ndim=4, ls=[0.8, 0.5, 0.7, 0.2], sym=False)


def _test_CG(grid_level, ndim, ls):
    W, base_lazy_tensor,  desired_lhs, kmat, Kinterp = help_setup(grid_level, ndim, ls)

    rhs = (Kinterp._matmul(desired_lhs)).detach()
    rhs_norm = torch.sqrt(torch.sum(rhs ** 2, 0)).detach()
    desired_lhs = desired_lhs/rhs_norm
    rhs = (Kinterp._matmul(desired_lhs)).detach()

    config.set_gpytorch_settings(cg_tol=1e-10, max_cg_iters=500)
    actual_lhs = Kinterp.inv_matmul(rhs)

    max_element_diff = np.max(np.abs((Kinterp._matmul(actual_lhs)).detach().numpy() - rhs.detach().numpy()))
    return max_element_diff


def test_CG1():
    med = _test_CG(grid_level=2, ndim=2, ls=[0.8, 0.9])
    np.testing.assert_array_less(med, MED_TOLERANCE)


if __name__ == '__main__':
    test_MVM5()
