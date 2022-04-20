import numpy as np
import pandas as pd


def simulate_bivariate_abpnl(n: int) -> dict:
    def f1(cause: np.array) -> np.array:
        return cause ** (-1) + 10 * cause

    def f2(arg: np.array) -> np.array:
        return arg ** 3

    x = np.random.uniform(0.1, 1.1, n)
    noise = np.random.uniform(0, 5, n)

    z = f1(x) + noise
    y = f2(z)
    df = pd.DataFrame({'x': x, 'y': y})
    sim_data = {'df': df, 'noise': noise}

    return sim_data
