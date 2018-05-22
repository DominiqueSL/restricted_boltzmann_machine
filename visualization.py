import matplotlib.pyplot as plt


def visualize_filters(weights):
    """
    Plot the weights (filters) in order to keep track of the training progress of the RBM
    Plotting the weights!
    :param weights: Matrix of the weights learned by the model
    """
    plt.imshow(weights)
    plt.show()


def visualize_neg_samples(reconstruction):
    """
    Visualize the negative samples in order to keep track of the training progress of the RBM
    :param reconstruction:
    """
    plt.imshow(reconstruction)
    plt.show()


def loss_plots(loss, epoch):
    """
    Make plots of the loss function over the number of epochs
    :param loss: List of floats corresponding to the loss of the model
    :param epoch: List of integers containing the epochs
    :return:
    """