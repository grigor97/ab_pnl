import argparse
from utils.utils import *
from src.models.model import *


def parse_args():
    parser = argparse.ArgumentParser("abpnl bivariate model")
    parser.add_argument('-c', '--config', type=str, default='configs/config.yml')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)

    n = config['params']['n']
    batch_size = config['params']['batch_size']
    lamb = config['params']['lamb']
    num_epochs = config['params']['num_epochs']
    num_trials = config['params']['num_trials']
    print(n, batch_size, lamb, num_epochs, num_trials)

    data = simulate_bivariate_abpnl(n)
    df = data['df']
    df = (df - df.mean()) / df.std()
    noise = data['noise']

    median_loss, losses = get_final_median_loss(df, batch_size, lamb, num_epochs, num_trials)

    median_loss_back, losses_back = get_final_median_loss(df[['x2', 'x1']], batch_size, lamb, num_epochs, num_trials)

    print('direction ->')
    print(median_loss)
    print(losses)
    
    print('direction <- ')
    print(median_loss_back)
    print(losses_back)

    if median_loss_back > median_loss:
        print("estimated direction is ->")
    else:
        print("estimated direction is <-")


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')