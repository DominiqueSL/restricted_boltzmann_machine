# Script to test the output of each function with a fixed dataset:
# training_data = np.array(
#     [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#      [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#      [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#      [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
#      [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
#      [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
# )
# with 4 hidden nodes
# use this line for testing the probability function:
# r._probability_hidden(training_data[0, :])
import numpy as np
import sys
import math


def test_probability_hid(probability, nr_nodes):
    """
    Test function for probability.
    Test dimensions of the visible vector
    Test if values in the vector are between 0 and 1
    Test if the values are equal to the calculated values by hand
    :param probability: numpy array of the probabilities of activation of the visible units
    :param nr_nodes: integer corresponding to the number of visible units
    :return: Error messages in case there is something wrong
    """
    if probability.shape[0] != nr_nodes:
        raise ValueError("Probability vector dimension does not correspond with number of hidden nodes")

    for i in range(nr_nodes):
        if probability[i] < 0 or probability[i] > 1:
            raise ValueError("Probability of visible unit must be within 0 and 1")

    if np.round(probability[0], 4) != 0.5088:
        raise ValueError("First index value does not correspond with actual value")

    if np.round(probability[1], 4) != 0.4996:
        raise ValueError("Second index value does not correspond with actual value")

    if np.round(probability[2], 4) != 0.5052:
        raise ValueError("Third index value does not correspond with actual value")

    if np.round(probability[3], 4) != 0.5089:
        raise ValueError("Fourth index value does not correspond with actual value")


def test_probability_vis(probability, nr_nodes):
    """
    Test function for probability.
    Test dimensions of the hidden vector
    Test if values in the vector are between 0 and 1
    Test if the values are equal to the calculated values by hand
    :param probability: numpy array of the probabilities of activation of the visible units
    :param nr_nodes: integer corresponding to the number of visible units
    :return: Error messages in case there is something wrong
    """
    if probability.shape[0] != nr_nodes:
        raise ValueError("Probability vector dimension does not correspond with number of visible nodes")

    for i in range(nr_nodes):
        if probability[i] < 0 or probability[i] > 1:
            raise ValueError("Probability of visible unit must be within 0 and 1")

    if np.round(probability[0], 4) != 0.5000:
        raise ValueError("First index value does not correspond with actual value")

    if np.round(probability[1], 4) != 0.5000:
        raise ValueError("Error, second index value does not correspond with actual value")

    if np.round(probability[2], 4) != 0.5000:
        raise ValueError("Third index value does not correspond with actual value")

    if np.round(probability[3], 4) != 0.5000:
        raise ValueError("Fourth index value does not correspond with actual value")


def test_sampling(samples, nr_nodes):
    """
    Test if sampling goes well
    :param samples: list of samples
    :param nr_nodes: integer corresponding to the number of nodes (hidden or visible)
    :return: Error message in case there is something wrong
    """
    if samples.shape[0] != nr_nodes:
        raise ValueError("Sampling vector is larger than number of available nodes")

    for i in range(nr_nodes):
        if (samples[i] > 1) or (samples[i] < 0):
            raise ValueError("Samples should be binary but are not")


def test_outer_prod_data(outer_product, weights):
    """
    Test if computation of the outer product is okay, which is required for the computation of the positive and negative
    gradient.
    :param outer_product: numpy array corresponding with the outer product of the dataset.
    :param weights: integer corresponding with the number of nodes (hidden or visible)
    :return: Error message in case something is wrong
    """
    if outer_product.shape[0] != weights.shape[0] or outer_product.shape[1] != weights.shape[1]:
        raise ValueError("Dimensions of the outer product for gradient computation is wrong (dimension mismatch)")
    # Maybe incorporate an outer product that is computed by hand
    # If data set stays the same then these number should also stay the same.


def test_cross_entropy(error_arr, prob_size):
    """
    Test if the cross entropy was computed correctly
    :param error_arr: numpy array containing the cross entropy of the model
    :param prob_size:
    :return:
    """
    for error in error_arr:
        if math.isnan(error):
            raise ValueError("Something went wrong during cross entropy computation. Answer should not be NaN")
        if math.isinf(error):
            raise ValueError("Something went wrong during cross entropy computation. Answer should not be Inf")

    if error_arr.shape[0] != prob_size:
        raise ValueError("Something went wrong during cross entropy computation. Error array is too large")
