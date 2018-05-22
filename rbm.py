import help_functions as hf
import numpy as np


class RBM:
    def __init__(self, training_data, num_visible, num_hidden):
        """
        Constructor of the restricted Boltzmann machine
        Takes care of the initialization of many variables
        :param training_data: Matrix containing all the data required for training of the network
        :param num_visible: Integer corresponding to the number of visible units in the network
        :param num_hidden: Integer corresponding to the number of hidden units in the network
        :param depth: Integer corresponding to the number of layers the network is deep
        """
        self.num_visible = num_visible
        self.hidden_nodes = np.ones(num_hidden)
        self.weights = np.random.normal(0.0, 0.01, (num_visible, num_hidden))
        self.bias_hid = np.zeros(num_hidden)
        self.bias_vis = self._bias_visible_init(training_data)
        self.vis_nodes_recon = np.zeros(num_visible)
        self.p_h_data = np.zeros(num_hidden)
        self.hid_nodes_recon = np.zeros(num_hidden)

    @staticmethod
    def _bias_visible_init(visible_units):
        """
        Compute the initial bias of the visible units
        :param visible_units: The activations of the visible units
        :return: tensor containing the bias of the visible units
        """
        proportion = np.divide(np.sum(visible_units, 0), visible_units.shape[0])
        denominator = np.subtract(np.ones(proportion.shape), proportion)
        return np.log(np.divide(proportion, denominator))

    def _probability_hidden(self, visible_nodes):
        """
        Computes the probability of turning on a hidden unit.
        :param visible_nodes: vector containing the activations of the visible nodes
        :return: Vector with the probability all the hidden nodes. Size of the vector should correspond to the number
        of hidden units.
        """
        return hf.sigmoid(self.bias_hid + np.dot(np.transpose(visible_nodes), self.weights))

    def _probability_visible(self, hidden_nodes):
        """
        Computes the conditional probability of turning on a visible unit.
        Implemented according to formula in Hinton's Practical guide to Restricted Boltzmann machine
        The probability is also just the update of the visible state.
        :param hidden_nodes: vector containing the activations of the hidden nodes
        :return: float containing probability of visible unit (single node)
        """
        return hf.sigmoid(self.bias_vis + np.dot(self.weights, np.transpose(hidden_nodes)))

    def _pos_gradient(self, vis_nodes):
        """
        Compute the positive gradient step, required for the contrastive divergence algorithm.
        :param vis_nodes: Vector containing the activation of the visible nodes
        :return: Vector with on each index the outer product between hidden and visible node
        :return: Vector with the hidden node activation
        """
        # Positive gradient: Visible => Hidden
        self.p_h_data = self._probability_hidden(vis_nodes)

        # Sample from this probability distribution
        sample_nodes = np.vectorize(self._sample_node)
        self.hid_nodes_recon = sample_nodes(self.p_h_data)

        # Data outer product
        return np.outer(vis_nodes, self.hid_nodes_recon), self.hid_nodes_recon

    def _neg_gradient(self, k=2):
        """
        Compute the outer product between hidden and visible nodes, which are obtained from reconstructions.
        :param k: Integer corresponding to the number of Gibbs sampling steps
        :return: Vector with on each index the outer product between hidden and visible nodes
        :return: Vector with all the visible node activations
        :return: Vector with all the hidden node activations
        """
        # Vectorize function for easy sampling
        sample_nodes = np.vectorize(self._sample_node)
        # Initialize the hidden nodes activation
        # hid_nodes_act = hid_nodes
        # Perform Gibbs sampling
        for step in range(k):
            # Negative gradient: Hidden => Visible (reconstruction of data)
            p_v = self._probability_visible(self.hid_nodes_recon)
            # Sample
            self.vis_nodes_recon = sample_nodes(p_v)
            # Reconstruct the hidden nodes from visible again
            p_h = self._probability_hidden(self.vis_nodes_recon)
            # Sample from this probability distribution
            self.hid_nodes_recon = sample_nodes(p_h)
        return np.outer(self.vis_nodes_recon, self.hid_nodes_recon)

    def _update_model_params(self, vis_nodes, lr=0.01, k=1):
        """
        Approximate the gradient using the contrastive divergence algorithm. This is required to compute the weight update,
        as well as other model parameter updates.
        :param vis_nodes vector containing the input data from the visible nodes
        :param lr: float corresponding to the learning rate of the model
        :param k: int corresponding to the number of iterations using CD

        :return: Reconstructed visible nodes are returned, as they are needed to compute the error
        """
        # Compute positive gradient
        pos_grad, hid_node_recon_data = self._pos_gradient(vis_nodes)

        # Iterate k number of times
        for i in range(k):
            neg_grad = self._neg_gradient()

        # Update
        self.weights += lr * (pos_grad - neg_grad)
        self.bias_hid += lr * (hid_node_recon_data - self.hid_nodes_recon)

    def empirical_probability(self, training_data):
        """
        Compute the empirical probability of the visible nodes
        :param training_data: Matrix containing the input data with the samples on the rows and features on columns
        :return: Vector containing the empirical probability of the visible nodes
        """
        return np.divide(np.sum(training_data, 0), training_data.shape[0])

    @staticmethod
    def _sample_node(prob):
        """
        Obtain unbiased sample of node. Sample from conditional probability. Needed to perform Gibbs sampling step.
        :param prob: float corresponding with the probability of a node being activated
        :return: binary number corresponding with the activation of the node
        """
        return np.random.binomial(1, prob)  # If you sample binomial once you sample Bernoulli

    def train(self, input_data, max_epochs=100, lr=0.01, error_threshold=0.1):
        """
        Train the restricted Boltzmann machine (RBM)
        :param input_data: Matrix containing samples and features
        :param max_epochs: Integer corresponding to the maximum number of iterations over one training batch
        :param lr: Float corresponding to the learning rate of the model
        :param error_threshold: Float, if error is below this value, model can stop training
        """
        epoch = 0
        error_square = float("inf")

        while epoch <= max_epochs and error_square > error_threshold:
            # First compute the empirical distribution of the visible nodes
            p_v = self.empirical_probability(input_data)
            # Vectorize function for easy sampling
            sample_nodes = np.vectorize(self._sample_node)
            v = sample_nodes(p_v)
            # Do contrastive divergence algorithm
            self._update_model_params(v, lr=lr)
            # Compute error
            error_square = hf.squared_recon_error(v, self.vis_nodes_recon)
            # error_cross = hf.cross_entropy_error(v, self.vis_nodes_recon)
            # print("Squared error: %.3f" % error_square + "\n")
            # # print("Cross entropy error: %.3f" % error_cross + "\n")
            # print("Epoch: " + str(epoch) + "\n")
            epoch += 1

        p_v = self._probability_visible(self.hid_nodes_recon)
        p_h = self._probability_hidden(self.vis_nodes_recon)
        print(p_v)
        print(p_h)

    def make_prediction(self):
        """
        Makes prediction of the state of the visible nodes
        :return:
        """
