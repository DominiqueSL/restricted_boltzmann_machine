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
        # self.bias_vis = self._bias_visible_init(training_data)
        self.bias_vis = np.zeros(num_visible)
        self.vis_nodes_recon = np.zeros(training_data.shape)
        self.p_h_data = np.zeros(num_hidden)
        self.hid_nodes_recon = np.zeros(num_hidden)

    def _energy(self, visible_nodes, hidden_nodes):
        return -np.dot(np.transpose(hidden_nodes), self.bias_hid) - np.dot(np.tranpose(hidden_nodes), self.weights,
                                                                           visible_nodes) - np.dot(
            np.transpose(self.bias_vis), visible_nodes)

    def _free_energy(self, weights, visible_nodes, bias_vis, bias_hid):
        """
        Free energy computation without the use of the partition function.
        :param visible_nodes: Vector corresponding to the activations of the visible units
        :return: Float corresponding to the free energy of a visible vector
        """
        input_hidden = self._probability_hidden(visible_nodes, weights, bias_hid)
        return - np.dot(visible_nodes, bias_vis) - np.sum(np.log(1 + np.exp(input_hidden)))

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

    @staticmethod
    def _probability_hidden(visible_nodes, weights, bias_hid):
        """
        Computes the probability of turning on a hidden unit.
        :param visible_nodes: vector containing the activations of the visible nodes
        :return: Vector with the probability all the hidden nodes. Size of the vector should correspond to the number
        of hidden units.
        """
        return hf.sigmoid(bias_hid + np.dot(np.transpose(visible_nodes), weights))

    @staticmethod
    def _probability_visible(hidden_nodes, weights, bias_vis):
        """
        Computes the conditional probability of turning on a visible unit.
        Implemented according to formula in Hinton's Practical guide to Restricted Boltzmann machine
        The probability is also just the update of the visible state.
        :param hidden_nodes: vector containing the activations of the hidden nodes
        :return: float containing probability of visible unit (single node)
        """
        return hf.sigmoid(bias_vis + np.dot(hidden_nodes, np.transpose(weights)))

    def _pos_gradient(self, vis_nodes, weights, bias_hid):
        """
        Compute the positive gradient step, required for the contrastive divergence algorithm.
        :param vis_nodes: Vector containing the activation of the visible nodes
        :return: Vector with on each index the outer product between hidden and visible node
        :return: Vector with the hidden node activation
        """
        # Positive gradient: Visible => Hidden
        self.p_h_data = self._probability_hidden(vis_nodes, weights, bias_hid)

        # Sample from this probability distribution
        sample_nodes = np.vectorize(self._sample_node)
        self.hid_nodes_recon = sample_nodes(self.p_h_data)

        # Data outer product
        return np.outer(vis_nodes, self.hid_nodes_recon), self.hid_nodes_recon

    def _neg_gradient(self, weights, bias_vis, bias_hid, k=2):
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
            p_v = self._probability_visible(self.hid_nodes_recon, weights, bias_vis)
            # Sample
            self.vis_nodes_recon = sample_nodes(p_v)
            # Reconstruct the hidden nodes from visible again
            p_h = self._probability_hidden(self.vis_nodes_recon, weights, bias_hid)
            # Sample from this probability distribution
            self.hid_nodes_recon = sample_nodes(p_h)
        return np.outer(self.vis_nodes_recon, self.hid_nodes_recon)

    def empirical_probability(self, training_data):
        """
        Compute the empirical probability of the visible nodes
        :param training_data: Matrix containing the input data with the samples on the rows and features on columns
        :return: Vector containing the empirical probability of the visible nodes
        """
        return np.divide(np.sum(training_data, 0), training_data.shape[0])

    def _update_model_params(self, vis_nodes, weights, bias_visible, bias_hidden,  lr=0.01, k=1):
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
        pos_grad, hid_node_recon_data = self._pos_gradient(vis_nodes, weights, bias_hidden)

        # Iterate k number of times
        for i in range(k):
            neg_grad = self._neg_gradient(weights, bias_visible, bias_hidden)

        # Update
        weights += lr * (pos_grad - neg_grad)
        bias_hidden += lr * (hid_node_recon_data - self.hid_nodes_recon)
        bias_visible += lr * (vis_nodes - self.vis_nodes_recon)
        return weights, bias_visible, bias_hidden

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
        counter = 0

        for epoch in range(max_iterations):
            # Split up the data into training (9 parts), validation (1 part)
            split_nr = int(input_data.shape[0] * split * 0.9)
            np.random.shuffle(input_data)
            train = input_data[0:split_nr, :]
            val = input_data[split_nr:, :]

            # Initialize the parameters
            weights = self.weights
            bias_vis = self.bias_vis
            bias_hid = self.bias_hid
            free_energy_train = np.zeros(train.shape[0])
            free_energy_val = np.zeros(val.shape[0])

            # Train RBM
            for i in range(train.shape[0]):
                # Do contrastive divergence algorithm
                weights, bias_vis, bias_hid = self._update_model_params(train[i, :], weights, bias_vis, bias_hid, lr=lr)
                free_energy_train[i] = self._free_energy(weights, train[i, :], bias_vis, bias_hid)
            # Validate RBM
            for j in range(val.shape[0]):
                free_energy_val[j] = self._free_energy(weights, val[j, :], bias_vis, bias_hid)

            # Update here if the validation free energy does not go up
            # print("Epoch: " + str(epoch) + " \n")

            if np.abs(np.average(free_energy_val) - np.average(free_energy_train)) < 0.1:
                self.weights = weights
                self.bias_hid = bias_hid
                self.bias_vis = bias_vis
                # print("Updated the model parameters" + str(counter))
                counter+=1


    def check_model(self):
        """
        Check the model by computing the partition function and see if the
        :return:
        """

    def make_prediction(self, sample_data):
        """
        Makes prediction of the state of the visible nodes
        :param sample_data: Sample of the class you want to predict. Predict hidden nodes
        :return: Nothing
        """
        p_h = self._probability_hidden(sample_data, self.weights, self.bias_hid)
        sample_nodes = np.vectorize(self._sample_node)
        h = sample_nodes(p_h)
        p_v = self._probability_visible(h, self.weights, self.bias_vis)
        v = sample_nodes(p_v)
        print(h)
        print(v)
