import numpy as np
import help_functions as hf


class RBM:
    def __init__(self, training_data, num_visible, num_hidden):
        """
        Constructor of the restricted Boltzmann machine
        Takes care of the initialization of many variables
        :param training_data: Matrix containing all the data required for training of the network
        :param num_visible: Integer corresponding to the number of visible units in the network
        :param num_hidden: Integer corresponding to the number of hidden units in the network
        """
        self.num_visible = num_visible
        self.hidden_nodes = np.ones(num_hidden)
        self.weights = np.random.normal(0.0, 0.01, (num_visible, num_hidden))
        self.bias_hid = np.zeros(num_hidden)
        self.bias_vis = self._bias_visible_init(training_data)
        self.vis_nodes_recon = np.zeros(training_data.shape)
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
        return hf.sigmoid(self.bias_vis + np.dot(hidden_nodes, np.transpose(self.weights)))

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
        # nodes = self.empirical_probability(vis_nodes)
        pos_grad, hid_node_recon_data = self._pos_gradient(vis_nodes)

        # Iterate k number of times
        for i in range(k):
            neg_grad = self._neg_gradient()

        # Update
        self.weights += lr * (pos_grad - neg_grad)
        self.bias_hid += lr * (hid_node_recon_data - self.hid_nodes_recon)
        self.bias_vis += lr * (vis_nodes - self.vis_nodes_recon)

    @staticmethod
    def _sample_node(prob):
        """
        Obtain unbiased sample of node. Sample from conditional probability. Needed to perform Gibbs sampling step.
        :param prob: float corresponding with the probability of a node being activated
        :return: binary number corresponding with the activation of the node
        """
        return np.random.binomial(1, prob)  # If you sample binomial once you sample Bernoulli

    def train(self, input_data, split=0.8, max_iterations=100, lr=0.01):
        """
        Train the restricted Boltzmann machine (RBM)
        :param input_data: Matrix containing samples and features
        :param split: Float corresponding with the
        :param max_iterations: Integer corresponding to the maximum number of iterations over one training batch
        :param lr: Float corresponding to the learning rate of the model
        """
        for epoch in range(max_iterations):
            # Split up the data into training (9 parts), validation (1 part)
            split_nr = int(input_data.shape[0] * split * 0.9)
            np.random.shuffle(input_data)
            train = input_data[0:split_nr, :]
            val = input_data[split_nr:, :]

            # Initialization
            reconstruction_cost = 0
            # Train RBM
            for i in range(train.shape[0]):
                # Do contrastive divergence algorithm
                self._update_model_params(train[i, :], lr=lr)
                reconstruction_cost += self._cost_cross_entropy(train[i, :])

            print("Epoch: " + str(epoch+1) + "\n")
            print("Reconstruction cost on training set: " + str(reconstruction_cost/train.shape[0]).format("%.5f") + "\n")

            reconstruct_cost_val = 0

            # Validate RBM
            for j in range(val.shape[0]):
                reconstruct_cost_val += self._cost_cross_entropy(val[j, :])

            print("Reconstruction cost on validation set: " + str(reconstruct_cost_val/val.shape[0]).format("%.5f") + "\n")

    def _cost_cross_entropy(self, data):
        """
        Compute cross-entropy to keep track of training of the model.
        :param data: numpy array containing the input data
        :return: float containing the reconstruction cost
        """
        y_n = hf.sigmoid(np.dot(self.hid_nodes_recon, np.transpose(self.weights)))
        cross_entropy = data * np.log(y_n) + (1 - data) * np.log(1 - y_n)
        return -cross_entropy

    def make_prediction(self, sample_data):
        """
        Makes prediction of the state of the visible nodes
        :param sample_data: Sample of the class you want to predict. Predict hidden nodes
        :return: Nothing
        """
        p_h = self._probability_hidden(sample_data)
        sample_nodes = np.vectorize(self._sample_node)
        h = sample_nodes(p_h)
        p_v = self._probability_visible(h)
        v = sample_nodes(p_v)
        print(h)
        print(v)