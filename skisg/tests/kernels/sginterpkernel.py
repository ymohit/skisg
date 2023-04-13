import torch
import numpy as np
import gpytorch as gp

from matplotlib import pyplot as plt

from skisg.config import SgBasisType, InterpType
from skisg.kernels.sginterpkernel import SparseGridInterpolationKernel


def main():

    ndim = 2
    npoints = 160 + 1

    basis = SgBasisType.MODIFIED
    interp_type = InterpType.SIMPLEX

    gl = 6

    ndimpoints = 5
    epsilon = 0
    x1s = np.linspace(0 + epsilon, 1 - epsilon, num=ndimpoints)
    x2s = np.linspace(0 + epsilon, 1 - epsilon, num=ndimpoints)
    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid
    X = np.vstack([x1.ravel(), x2.ravel()]).T
    npoints = X.shape[0]
    func = lambda x:  np.sin(4 *np.pi *(x[:, 0] + x[:, 1]))

    base_covar_module = gp.kernels.RBFKernel()
    covar_module = SparseGridInterpolationKernel(
        base_kernel=base_covar_module,
        grid_level=gl,
        ndim=ndim,
        umin=-0.05,
        umax=1.05,
        comb=True,
        basis=SgBasisType.MODIFIED,
        interp_type=interp_type
    )
    X = torch.from_numpy(X)

    interp_kernel = covar_module.forward(x1=X, x2=X)
    eye_matrix = torch.eye(X.shape[0]).to(dtype=torch.float64)
    interp_matrix = interp_kernel.matmul(eye_matrix)
    actual = interp_matrix.detach().cpu().numpy()

    base_covariance_kernel = base_covar_module(X).evaluate()
    expected = base_covariance_kernel.detach().cpu().numpy()

    # print("True kernel ...", base_covariance_kernel.detach().cpu().numpy().shape)
    # plt.imshow(expected)
    # plt.colorbar()
    # plt.show()
    #
    # print("Interpolated kernel ...")
    # plt.imshow(actual)
    # plt.colorbar()
    # plt.show()

    print("Diff ...")
    plt.imshow(np.abs(expected - actual))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
