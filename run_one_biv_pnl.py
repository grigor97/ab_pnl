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

    input_dim = df.shape[1] - 1

    train, test = train_test_split(df, test_size=0.1, random_state=10, shuffle=True)

    train = np.array(train)
    test = np.array(test)

    train = MyDataset(train)
    test = MyDataset(test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    train_loss_avgs, test_loss_avgs, min_loss = train_model(train_loader, test_loader,
                                                            lamb, num_epochs, input_dim)

    plot_abpnl_bivariate_losses(train_loss_avgs, test_loss_avgs)

    print('min loss')
    print(min_loss)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')