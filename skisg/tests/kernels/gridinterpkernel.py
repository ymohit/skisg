import torch
import numpy as np
import gpytorch as gp

from skisg.kernels.gridinterpkernel import ModifiedGridInterpolationKernel
from matplotlib import pyplot as plt
from skisg.config import set_seeds

set_seeds()


def _data(n=2):
    train_x = torch.zeros(pow(n, 2), 2)
    for i in range(n):
        for j in range(n):
            train_x[i * n + j][0] = float(i) / (n - 1)
            train_x[i * n + j][1] = float(j) / (n - 1)
    return train_x


def _setup(grid_size=27, num_dims=2):

    spacing = 1.0/grid_size

    if num_dims == 2:
        grid_bounds = ((0.0-spacing, 1.0+spacing), (0.0-spacing, 1.0+spacing))
    elif num_dims == 1:
        grid_bounds = ((0.0 - spacing, 1.0 + spacing),)
    else:
        raise NotImplementedError

    covar_module1 = gp.kernels.ScaleKernel(
        gp.kernels.GridInterpolationKernel(
            gp.kernels.RBFKernel(), grid_size=grid_size, grid_bounds=grid_bounds, num_dims=num_dims
        )
    )

    covar_module2 = gp.kernels.ScaleKernel(
        ModifiedGridInterpolationKernel(
            gp.kernels.RBFKernel(), grid_size=grid_size, grid_bounds=grid_bounds, num_dims=num_dims,
            adjust_boundary=False,
        )
    )

    return covar_module1, covar_module2


h_plt_tensor = lambda X: plt.imshow(X.detach().numpy())


def plt_tensor(X):
    try:
        X = X.evaluate()
    except AttributeError:
        pass

    h_plt_tensor(X)
    plt.colorbar()
    plt.show()


def test_grid_kernel_matrix():

    for n_ in [2, 4, 10]:
        for gs in [10, 21, 30]:
            train_x = _data(n=n_)
            desired, actual = _setup(grid_size=gs)
            desired_covar = desired.forward(train_x, train_x)
            actual_covar = actual.forward(train_x, train_x)
            d_c_base = desired_covar.base_lazy_tensor.evaluate().detach().numpy()
            a_c_base = actual_covar.base_lazy_tensor.evaluate().detach().numpy()
            np.testing.assert_almost_equal(actual=a_c_base, desired=d_c_base)


def test_interp_kernel_matrix_1d():
    def obtain_W(covar):
        self = covar
        return self._sparse_left_interp_t(self.left_interp_indices, self.left_interp_values).to_dense().transpose(-1, -2)

    train_x = torch.rand(2).reshape(-1, 1)
    desired, actual = _setup(grid_size=6, num_dims=1)
    desired_covar = desired.forward(train_x, train_x)
    actual_covar = actual.forward(train_x, train_x)

    print(train_x)
    print(desired.base_kernel.grid)
    print(actual.base_kernel.grid)

    desired_left_interp = obtain_W(desired_covar)
    actual_left_interp = actual_covar.left_interp_tensor.to_dense()

    # plt_tensor(desired_left_interp)
    # plt_tensor(actual_left_interp)

    # torch.set_printoptions(precision=2)
    print(desired_left_interp)
    print(actual_left_interp)


def test_interp_kernel_matrix():
    def obtain_W(covar):
        self = covar
        return self._sparse_left_interp_t(self.left_interp_indices, self.left_interp_values).to_dense().transpose(-1, -2)

    train_x = _data(n=2)
    desired, actual = _setup(grid_size=5)
    desired_covar = desired.forward(train_x, train_x)
    actual_covar = actual.forward(train_x, train_x)

    desired_left_interp = obtain_W(desired_covar)
    actual_left_interp = actual_covar.left_interp_tensor.to_dense()

    plt_tensor(desired_left_interp)
    plt_tensor(actual_left_interp)

    # d_c_base = desired_covar.base_lazy_tensor.evaluate().detach().numpy()
    # a_c_base = actual_covar.base_lazy_tensor.evaluate().detach().numpy()
    # np.testing.assert_almost_equal(actual=a_c_base, desired=d_c_base)
    #

if __name__ == '__main__':
    test_interp_kernel_matrix_1d()