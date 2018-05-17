import tensorflow as tf
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
