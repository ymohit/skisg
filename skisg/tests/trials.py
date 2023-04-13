import torch
import scipy.sparse
import numpy as np
import math

import gpytorch as gp
from skisg.config import SgBasisType, InterpType
from skisg.interp.sparse.sglocations import get_sg_points_nd
from skisg.interp.sparse.sgindices import compute_LI
from skisg.interp.sginterp import SparseInterpolation
from skisg.interp.sparse.nbhors import compute_comb_B_diag
from gpytorch.lazy import MatmulLazyTensor
from skisg.config import set_seeds
import skisg.config as config


def spinterpolate_interpolation(func, X, gl, ndim, umin, umax,
                  kind=InterpType.LINEAR, basis=SgBasisType.NAIVE, comb=True,
                  shifted=config.SgShifted.ZERO
                    ):

    # evaluating function on grid
    sg_locs = get_sg_points_nd(gl, ndim, umin=umin, umax=umax, basis=basis, comb=True, ordered=False)
    order = compute_LI(gl, ndim, basis=basis, comb=comb, rmat=False)
    sg_locs = sg_locs[order, :]
    f_sg = torch.from_numpy(func(sg_locs))
    x_target = torch.from_numpy(X) if not torch.is_tensor(X) else X
    phi = SparseInterpolation().sparse_interpolate(
        grid_level=gl,
        ndim=ndim, umin=umin, umax=umax,
        x_target=x_target,
        comb=comb,
        interp_type=kind,
        basis=basis,
        device=x_target.device,
        shifted=shifted,
    )
    if shifted == config.SgShifted.ZERO:
        B = compute_comb_B_diag(gl, ndim, basis=basis, device=x_target.device, dtype=x_target.dtype)
    else:
        if shifted == config.SgShifted.ONE:
            B_diag = 1/ndim
        elif shifted == config.SgShifted.TWO:
            B_diag = 1 / (math.comb(ndim, 2) + ndim)
        else:
            raise NotImplementedError
        B_diag = config.np2torch(B_diag*np.ones(phi.shape[1]), device=phi.device, dtype=phi.dtype)
        B = gp.lazy.DiagLazyTensor(B_diag)

    W = MatmulLazyTensor(phi, B)
    f_h = W.matmul(f_sg).detach().numpy()  # .reshape(-1)
    return f_h, phi.shape[1]


def err(f1, f2):
    return np.mean(np.abs(f1-f2))


def run_experiment(ndim=5, max_gl=3, shifted=config.SgShifted.ZERO):

    assert max_gl < ndim

    ntest = 200
    Xtest = np.random.rand(ntest, ndim)
    umin = -0.01
    umax = 1.01

    sg_func = lambda X: np.sin(np.sum(X, axis=1))
    f_t = sg_func(Xtest)

    for gl in range(2, max_gl+1):
        f_h, numpoints = spinterpolate_interpolation(
            sg_func, Xtest, gl, ndim, umin, umax, kind=InterpType(2), basis=SgBasisType(1),
            shifted=shifted
        )
        sg_error = err(f_h, f_t)
        print("\nErrors: ", gl, sg_error, err(np.zeros_like(f_t), f_t))


if __name__ == '__main__':
    set_seeds(1)
    #run_experiment(ndim=6, max_gl=4)
    run_experiment(ndim=8, max_gl=5, shifted=config.SgShifted.ONE)
    run_experiment(ndim=8, max_gl=4, shifted=config.SgShifted.TWO)
