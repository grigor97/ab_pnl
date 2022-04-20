import yaml

import matplotlib.pyplot as plt


def load_cfg(yaml_file_path):
    """
    Loads a yaml config file
    :param yaml_file_path: path of yaml file
    :return: config corresponding the path
    """
    with open(yaml_file_path, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    return cfg


def plot_abpnl_bivariate_losses(train_loss_avgs, test_loss_avgs):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(train_loss_avgs, label='train')
    ax.plot(test_loss_avgs, label='test')
    ax.legend()

    plt.savefig('train_test_loss.png')
    plt.show()
