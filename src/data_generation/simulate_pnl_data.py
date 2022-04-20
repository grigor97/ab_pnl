import numpy as np
import pandas as pd


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
