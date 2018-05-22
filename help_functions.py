import numpy as np
import os


def read_data(filename):
    """
    Reads data and converts to tensor
    :param filename: String containing filename
    :return: Tensor containing input data
    """
    cwd = os.getcwd()

    # Check if file exists
    if not os.path.isfile(cwd + filename):
        print("File does not exist")

    # Read file
    if filename.find(".csv") != -1:
        return np.genfromtxt(cwd + filename, delimiter=',')
    else:
        return np.loadtxt(cwd + filename)


def squared_recon_error(visible_nodes, recon_visible_nodes):
    """
    Compute the reconstruction error between the original data and the predictions, by computing the squared error
    Easy way out is to just compute the root mean squared error
    :param visible_nodes: Vector containing the original data
    :param recon_visible
    :return: error
    """
    # Simply take the squared error between the original input data and the reconstructed data
    norm = np.linalg.norm(visible_nodes - recon_visible_nodes)
    return np.square(norm)


def cross_entropy_error(visible_nodes, recon_visible_nodes):
    """
    Compute the reconstruction error using the original data and the predictions, using the cross-entropy
    Hinton's practical guide to RBM noted that this is most appropriate for the restricted Boltzmann machine combined
    with the contrastive divergence algorithm.
    :param visible_nodes: vector containing the visible nodes
    :param recon_visible_nodes: vector containing the reconstruction of the visible nodes
    :return:
    """
    return -np.dot(visible_nodes, np.log(recon_visible_nodes))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))