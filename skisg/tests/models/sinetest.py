import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import skisg.config as config

from experiments.setups import setup_ski
from experiments.setups import setup_sparse_grid
from tqdm import tqdm


count_params = lambda model: sum(p.numel() for p in model.parameters())


def train(model, mll, train_x, train_y, training_iterations=30):

    model.train()
    model.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    print('Tuning hyper-parameters ...', "Num parameters: ", count_params(model))

    for i in tqdm(range(training_iterations)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        # print(i, loss.item())
        # print(i, list(model.named_hyperparameters()))
        # for param_name, param in model.named_parameters():
        #     print('Parameter name: ', param_name,  'value = ', param.detach().numpy())
    return


def get_test(dtype, n=10):
    # Generate nxn grid of test points spaced on a grid of size 1/(n-1) in [0,1]x[0,1]

    test_x = torch.zeros(int(pow(n, 2)), 2)
    for i in range(n):
        for j in range(n):
            test_x[i * n + j][0] = float(i) / (n-1)
            test_x[i * n + j][1] = float(j) / (n-1)
    # Calc absolute error
    test_y_actual = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
    test_x = test_x.to(dtype=dtype)
    test_y_actual = test_y_actual.to(dtype=dtype)
    return test_x, test_y_actual


def eval(model, dtype, trip_boundary=False):
    # Set model and likelihood into evaluation mode
    model.eval()
    model.likelihood.eval()

    n = 10
    test_x, test_y_actual = get_test(dtype, n=10)

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cholesky_size(1):
        observed_pred = model.likelihood(model(test_x))
        pred_labels = observed_pred.mean.view(n, n)

    delta_y = torch.abs(pred_labels - test_y_actual).detach().numpy()

    if trip_boundary:
        delta_y = delta_y[1:-1, 1:-1]

    errors = delta_y.flatten()

    print("Error:", np.mean(errors), ' +/- ', np.std(errors), ', Zero pred error: ',
          np.mean(torch.abs(test_y_actual).detach().numpy()))
    return pred_labels, test_y_actual, delta_y, np.mean(errors)


# Define a plotting function
def ax_plot(f, ax, y_labels, title):
    im = ax.imshow(y_labels)
    ax.set_title(title)
    f.colorbar(im)
    plt.show()


def get_data(dtype):
    n = 40
    train_x = torch.zeros(pow(n, 2), 2)
    for i in range(n):
        for j in range(n):
            train_x[i * n + j][0] = float(i) / (n - 1)
            train_x[i * n + j][1] = float(j) / (n - 1)

    # True function is sin( 2*pi*(x0+x1))
    train_y = torch.sin((train_x[:, 0] + train_x[:, 1]) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)

    return train_x.to(dtype=dtype), train_y.to(dtype=dtype)


def run_sinetest(model, train_x, train_y, dtype, plot=False, trip_boundary=False):

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    train(model=model, mll=mll, train_x=train_x, train_y=train_y)

    pred_labels, test_y_actual, delta_y, error = eval(model, dtype, trip_boundary=trip_boundary)

    # Plot our predictive means
    # f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
    # ax_plot(f, observed_ax, pred_labels, 'Predicted Values (Likelihood)')
    #
    # # Plot the true values
    # f, observed_ax2 = plt.subplots(1, 1, figsize=(4, 3))
    # ax_plot(f, observed_ax2, test_y_actual, 'Actual Values (Likelihood)')

    # Plot the absolute errors
    if plot:
        f, observed_ax3 = plt.subplots(1, 1, figsize=(4, 3))
        ax_plot(f, observed_ax3, delta_y, 'Absolute Error Surface')

    return error


def get_default_config():
    boundary_slack = 0.2

    kernel_type = config.KernelsType.RBFKERNEL
    min_noise = 1e-4
    device = config.get_device()
    bypass_covar = False
    use_modified = True
    return boundary_slack, kernel_type, min_noise, device, bypass_covar, use_modified


def test_ski_linear():
    grid_level = 5
    dtype = config.dtype(use_torch=True)
    train_x, train_y = get_data(dtype)

    boundary_slack, kernel_type, min_noise, device, bypass_covar, use_modified = get_default_config()
    interp_type = config.InterpType.LINEAR
    boundary_slack = 0.1

    model = setup_ski(
        train_x, boundary_slack, train_y, grid_level,
        kernel_type, min_noise, device, dtype, interp_type, bypass_covar, use_modified
    )

    error = run_sinetest(model, train_x, train_y, dtype)

    assert error < 1e-2


def test_ski_cubic():
    grid_level = 5
    dtype = config.dtype(use_torch=True)
    train_x, train_y = get_data(dtype)

    boundary_slack, kernel_type, min_noise, device, bypass_covar, use_modified = get_default_config()
    use_modified = False
    interp_type = config.InterpType.CUBIC
    boundary_slack = 0.1

    model = setup_ski(
        train_x, boundary_slack, train_y, grid_level,
        kernel_type, min_noise, device, dtype, interp_type, bypass_covar, use_modified
    )

    error = run_sinetest(model, train_x, train_y, dtype)

    assert error < 1e-2


def test_ski_simplex():
    grid_level = 7
    dtype = config.dtype(use_torch=True)
    train_x, train_y = get_data(dtype)

    boundary_slack, kernel_type, min_noise, device, bypass_covar, use_modified = get_default_config()
    interp_type = config.InterpType.SIMPLEX

    model = setup_ski(
        train_x, boundary_slack, train_y, grid_level,
        kernel_type, min_noise, device, dtype, interp_type, bypass_covar, use_modified
    )

    error = run_sinetest(model, train_x, train_y, dtype)
    assert error < 1e-2


def test_sgki_linear():
    grid_level = 7
    dtype = config.dtype(use_torch=True)
    train_x, train_y = get_data(dtype)

    boundary_slack, kernel_type, min_noise, device, bypass_covar, use_modified = get_default_config()
    interp_type = config.InterpType.LINEAR
    boundary_slack = 0.1
    basis_type = config.SgBasisType.MODIFIED
    val_x, val_y = train_x, train_y

    test_x, test_y_actual = get_test(dtype, n=10)

    model = setup_sparse_grid(train_x, val_x, test_x, boundary_slack, train_y, grid_level,
            interp_type, basis_type, kernel_type, min_noise, device, dtype
    )
    error = run_sinetest(model, train_x, train_y, dtype)

    assert error < 1e-2, error


def test_sgki_cubic():

    grid_level = 7
    dtype = config.dtype(use_torch=True)
    train_x, train_y = get_data(dtype)

    boundary_slack, kernel_type, min_noise, device, bypass_covar, use_modified = get_default_config()
    interp_type = config.InterpType.CUBIC
    boundary_slack = 0.15
    basis_type = config.SgBasisType.MODIFIED
    val_x, val_y = train_x, train_y

    test_x, test_y_actual = get_test(dtype, n=10)

    model = setup_sparse_grid(train_x, val_x, test_x, boundary_slack, train_y, grid_level,
            interp_type, basis_type, kernel_type, min_noise, device, dtype
    )
    error = run_sinetest(model, train_x, train_y, dtype)
    assert error < 1e-2, error


def test_sgki_simplex():
    grid_level = 7
    dtype = config.dtype(use_torch=True)
    train_x, train_y = get_data(dtype)

    boundary_slack, kernel_type, min_noise, device, bypass_covar, use_modified = get_default_config()
    interp_type = config.InterpType.CUBIC
    boundary_slack = 0.2
    basis_type = config.SgBasisType.MODIFIED
    val_x, val_y = train_x, train_y

    test_x, test_y_actual = get_test(dtype, n=10)

    model = setup_sparse_grid(train_x, val_x, test_x, boundary_slack, train_y, grid_level,
                              interp_type, basis_type, kernel_type, min_noise, device, dtype)

    error = run_sinetest(model, train_x, train_y, dtype)

    assert error < 1e-2


if __name__ == '__main__':
    # test_ski_linear()
    # test_ski_cubic()
    # test_ski_simplex()
    # test_sgki_linear()
    # test_sgki_cubic()
    test_sgki_simplex()
    #help_sgki_cubic()
