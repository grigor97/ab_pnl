import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from utils.utils import *


def simulate_bivariate_abpnl(n: int) -> dict:
    def f1(x: np.array) -> np.array:
        return x ** (-1) + 10 * x

    def f2(z: np.array) -> np.array:
        return z ** 3

    x = np.random.uniform(0.1, 1.1, n)
    noise = np.random.uniform(0, 5, n)

    z = f1(x) + noise
    y = f2(z)
    df = pd.DataFrame({'x1': x, 'x2': y})
    sim_data = {'df': df, 'noise': noise}

    return sim_data


def simulate_multivariate_abpnl(n: int, d: int):
    # weighted sum of Gaussian processes according to the paper. Note that in the paper
    # it is not mentioned how to choose weights
    def f1(x):
        gpr = GaussianProcessRegressor()
        rand_seed = np.random.randint(0, 1000000)
        gp_vals = gpr.sample_y(x, n_samples=x.shape[1], random_state=rand_seed)
        w = np.random.uniform(0, 100, x.shape[1])
        w /= w.sum()

        f1_x = gp_vals @ w
        return f1_x

    # sigmoid function according to the paper. In the paper it is written
    # weighted sum of the sigmoid functions but we have univariate input here, which
    # means something is not correct in the paper. I think it should be just sigmoid
    # FIXME normalize
    def f2(z):
        return 1 / (1 + np.exp(-x))

    A = gen_directed_erdos_reyni_graph(d)

    X_vals = np.zeros(shape=(n, d))
    for i in range(d):
        parents_idx = np.where(A[:, i] == 1)[0]
        num_parents = len(parents_idx)
        if num_parents == 0:
            x = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=n)
        else:
            noise = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=n)

            parents = X_vals[:, np.where(A[:, i] == 1)[0]]
            z = f1(parents) + noise
            x = f2(z)

        X_vals[:, i] = x

    return A, X_vals
