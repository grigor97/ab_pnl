import argparse
from src.models.model import *


def parse_args():
    parser = argparse.ArgumentParser("abpnl multivariate model")
    parser.add_argument('-c', '--config', type=str, default='configs/config.yml')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)

    n = config['mult_params']['n']
    d = config['mult_params']['d']

    batch_size = config['params']['batch_size']
    lamb = config['params']['lamb']
    num_epochs = config['params']['num_epochs']
    num_trials = config['params']['num_trials']

    print(n, d, batch_size, lamb, num_epochs, num_trials)

    print("start data simulation")
    A, X_vals = simulate_multivariate_abpnl(n, d)
    print("end data simulation")
    plot_graph_from_adj_matrix(A)

    print('start training')

    order = get_order_mult(X_vals, batch_size, lamb, num_epochs, num_trials)

    print("causal order is ", order)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')