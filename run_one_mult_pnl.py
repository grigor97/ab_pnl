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

    train, test = train_test_split(X_vals, test_size=0.5, random_state=10, shuffle=True)

    train = MyDataset(train)
    test = MyDataset(test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    input_dim = X_vals.shape[1] - 1
    print('start training')
    train_loss_avgs, test_loss_avgs, min_loss = train_mult_model(train_loader, test_loader,
                                                                 lamb, num_epochs, input_dim)

    plot_abpnl_bivariate_losses(train_loss_avgs, test_loss_avgs)

    print('min loss')
    print(min_loss)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')