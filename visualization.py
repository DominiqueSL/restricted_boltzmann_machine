import matplotlib.pyplot as plt


def visualize_filters(weights):
    """
    Plot the weights (filters) in order to keep track of the training progress of the RBM
    Plotting the weights!
    :param weights: Matrix of the weights learned by the model
    """
    plt.imshow(weights)
    plt.show()


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


def log_likelihood_plots(epoch, likelihood_train, likelihood_val):
    """
    Visualize the log likelihood in order to
    :param epoch: list with all the iterations
    :param likelihood_train: numpy array corresponding to the likelihood of the model per iteration for training set
    :param likelihood_val: numpy array corresponding to the likelihood of the model per iteration for validation set
    """
    plt.figure()
    plt.plot(epoch, likelihood_train)
    plt.plot(epoch, likelihood_val)
    plt.xlabel("Iterations")
    plt.ylabel("Log-likelihood")
    plt.show()


def loss_plots(epoch, loss_train, loss_val):
    """
    Visualize the loss computed using the cross entropy on the train, validation and test set
    :param epoch: list with the number of iterations
    :param loss_train: numpy array with average loss of training set computed per iteration
    :param loss_val: numpy array with average loss of validation set computed per iteration
    """
    plt.figure()
    plt.plot(epoch, loss_train)
    plt.plot(epoch, loss_val)
    plt.xlabel("Iterations")
    plt.ylabel("Reconstruction error")
    plt.show()