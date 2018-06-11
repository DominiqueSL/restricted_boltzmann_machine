import numpy as np
import os
import pandas as pd
import glob
import h5py


def read_data(filename, heading=None):
    """
    Reads data and converts to matrix.
    Incorporate the .glob in this function, in order to read in large amounts of data
    :param filename: String containing filename
    :param heading: Boolean describing if there is a heading in the dataset
    :return: numpy array containing input data. Default behavior is None.
    """
    cwd = os.getcwd()

    # Check if file exists
    if not os.path.isfile(cwd + filename):
        raise FileNotFoundError("File does not exist/File not found")

    # Read file
    if filename.find(".csv") != -1:
        # Still need to test this function!!!
        names = glob.glob("*.csv")
        data = np.empty((1,1))
        for name in names:
            data = pd.DataFrame.as_matrix(pd.read_csv(cwd + name, header=heading))
        return data
    else:
        return np.loadtxt(cwd + filename)


def sum_squared_recon_error(visible_nodes, recon_visible_nodes):
    """
    Compute the reconstruction error between the original data and the predictions, by computing the squared error
    Easy way out is to just compute the root mean squared error
    :param visible_nodes: numpy array that contains the original data
    :param recon_visible: numpy array that contains the reconstruction of the data
    :return: float corresponding with the sum of squared error
    """
    # Simply take the squared error between the original input data and the reconstructed data
    norm = np.linalg.norm(visible_nodes - recon_visible_nodes)
    return np.sqrt(np.square(norm))


def cross_entropy(data, p_n):
    """
    Compute cross-entropy to keep track of training of the model.
    :param data: numpy array containing the input data (visible nodes)
    :param p_n: numpy array containing the reconstruction probability of visible nodes
    :return: numpy array containing the cross entropy for all the visible nodes.
    """
    cross_entropy = data * np.log(p_n) + (1 - data) * np.log(1 - p_n)
    return -cross_entropy


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def write_h5py(array, filename):
    """
    Function that writes array to h5py files
    :param array: numpy array that needs to be saved in h5py file
    :param filename: string corresponding to the file name
    """
    with h5py.File(filename, "w") as f:
        f.create_dataset("image", data=array, dtype=array.dtype)
