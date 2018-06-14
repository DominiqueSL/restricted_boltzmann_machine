import numpy as np
import os
import pandas as pd
import h5py
import gzip


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


def load_hdf5(infile):
    """
    Function that loads h5py files
    :param infile: string with the filename that is to be loaded
    :return: numpy array with the data in the h5 file.
    """
    with h5py.File(infile, "r") as f:  #"with" close the file after its nested commands
        return f["dataset"][()]


def write_h5py(array, filename):
    """
    Function that writes array to an hdf5 output file
    :param array: numpy array corresponding with the array that has to be written to hdf5 file
    :param filename: string corresponding with the name of the output file
    """
    with h5py.File(filename + ".hdf5", "w") as f:
        f.create_dataset("dataset", data=array, dtype=array.dtype)


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    IMAGE_SIZE = 28
    with gzip.open(filename, 'rb') as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, IMAGE_SIZE*IMAGE_SIZE)
    return np.round((data / 255), 0)
