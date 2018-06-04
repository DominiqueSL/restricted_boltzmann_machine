import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import os


def visualize_filters(weights):
    """
    Plot the weights (filters) in order to keep track of the training progress of the RBM
    Plotting the weights!
    :param weights: Matrix of the weights learned by the model
    """
    plt.imshow(weights)
    plt.show()


def visualize_hidden_prob_activation(hid_probability, out_name):
    """
    Function that visualizes the activation probabilities of the hidden nodes.
    At the beginning of training progress, it should a gray image. After training for some time, there should be more
    prominent black and white areas.
    The hidden neurons are given along the columns.
    The samples are shown given along the rows.
    White pixels represent a probability of 0, while black pixels represent a probability of 1.
    :param hid_probability: numpy array containing the probability of the hidden nodes
                            Note that this cannot be of a single sample only. It must be a 2D array in order to be
                            converted to an image.
    :param out_name: string corresponding to the name of the output image file
    """
    image = Image.fromarray(np.multiply(hid_probability, 255).astype(np.uint8))
    cwd = os.getcwd()
    dir_name = "./Probability_activation//"
    if not os.path.exists(cwd + dir_name):
        os.mkdir(dir_name)

    image.save(cwd + dir_name + "/" + out_name + ".png")


def visualize_neg_samples(reconstruction, image=False):
    """
    Visualize the negative samples in order to keep track of the training progress of the RBM
    :param reconstruction:
    """
    if image:
        plt.imshow(reconstruction)
        plt.show()
    else:
        print(reconstruction)


def model_param_visualization(weights, v_bias, hid_bias, dweights, dv_bias, dhid_bias, name_out):
    """
    Plot histograms of the the parameters the model has to learn.
    Under normal conditions, the histograms should roughly look like Gaussian distributions, during training.
    Mean magnitudes should be smaller than its corresponding upper plot, by factor 10^2 or 10^4.
    If the change in weights is too small, then the learning rate can be increased
    If the change in the weight is too large (and grow to infinity), then the learning rate is too high.
    Sometimes it can diverge into two bifurcate Gaussian distributions.
    :param weights: numpy array corresponding with the weights learnt by the model
    :param v_bias: numpy array corresponding with the bias for the visible units learnt by the model
    :param hid_bias: numpy array corresponding with the bias for the hidden units learnt by the model
    :param dweights: numpy array corresponding with the update for the weights learnt by the model
    :param dv_bias: numpy array corresponding with the update for the visible bias learnt by the model
    :param dhid_bias: numpy array corresponding with the update for the hidden bias learnt by the model
    :param name_out: string corresponding with the name of the output files
    """
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    fig = plt.figure()

    ax_1 = fig.add_subplot(231)
    # Visualize the weights, visible bias as well as hidden bias
    flat_weights = weights.flatten()
    ax_1.hist(flat_weights)
    ax_1.set_title('mm = %.6g' % np.mean(np.fabs(flat_weights)))

    ax_2 = fig.add_subplot(232)
    ax_2.hist(v_bias)
    ax_2.set_title('mm = %.6g' % np.mean(np.fabs(v_bias)))

    ax_3 = fig.add_subplot(233)
    ax_3.hist(hid_bias)
    ax_3.set_title('mm = %.6g' % np.mean(np.fabs(hid_bias)))

    # Visualize the update of the weights, visible bias as well as hidden bias
    ax_4 = fig.add_subplot(234)
    flat_update_weights = dweights.flatten()
    ax_4.hist(flat_update_weights)
    ax_4.set_title('mm = %.6g' % np.mean(np.fabs(flat_update_weights)))

    ax_5 = fig.add_subplot(235)
    ax_5.hist(dv_bias)
    ax_5.set_title('mm = %.6g' % np.mean(np.fabs(dv_bias)))

    ax_6 = fig.add_subplot(236)
    ax_6.hist(dhid_bias)
    ax_6.set_title('mm = %.6g' % np.mean(np.fabs(dhid_bias)))

    # Save the file
    cwd = os.getcwd()
    dir_name = "./Model_param_histograms//"
    if not os.path.exists(cwd + dir_name):
        os.mkdir(dir_name)

    fig.savefig(cwd + dir_name + name_out + ".png")


def loss_plots(epoch, loss_train, loss_val):
    """
    Visualize the loss computed using the cross entropy on the train, validation and test set
    :param epoch: list with the number of iterations
    :param loss_train: numpy array with average loss of training set computed per iteration
    :param loss_val: numpy array with average loss of validation set computed per iteration
    """
    plt.figure(0)
    plt.plot(epoch, loss_train, "r-")
    plt.plot(epoch, loss_val, "b-")

    plt.xlabel("Iterations")
    plt.ylabel("Reconstruction error")
    red_patch = mpatches.Patch(color='red', label='Training data')
    blue_patch = mpatches.Patch(color='blue', label='Validation data')
    plt.legend(handles=[red_patch, blue_patch])
    plt.legend()
    plt.show()