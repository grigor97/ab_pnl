import yaml

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


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
    ax.plot(train_loss_avgs[50:], label='train')
    ax.plot(test_loss_avgs[50:], label='test')
    ax.legend()

    plt.savefig('train_test_loss.png')
    plt.show()


def gen_directed_erdos_reyni_graph(d):
    edge_prob = 2 / (d - 1)

    A = np.random.binomial(1, edge_prob, size=d ** 2).reshape((d, d))
    A = np.triu(m=A, k=1)

    return A


def plot_graph_from_adj_matrix(A):
    d = A.shape[0]
    labels = dict(zip(range(d), np.array(range(d)) + 1))
    rows, cols = np.where(A == 1)
    edges = zip(rows, cols)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    G.add_edges_from(edges)
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, node_size=2000, labels=labels, with_labels=True)
    plt.savefig('generated_causal_graph.png')
    # plt.show()
