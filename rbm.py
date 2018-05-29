import numpy as np
import help_functions as hf
import test_individual_function as test_f
import visualization as vis
import matplotlib.pyplot as plt


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
        self.num_hidden = num_hidden
        self.hidden_nodes = np.ones(num_hidden)
        np.random.seed(0)
        self.weights = np.random.normal(0.0, 0.01, (num_visible, num_hidden))
        self.bias_hid = np.zeros(num_hidden)
        # self.bias_vis = self._bias_visible_init(training_data)
        self.bias_vis = np.zeros(num_visible)
        self.vis_nodes_recon = np.zeros(training_data.shape[1])
        self.p_h_data = np.zeros(num_hidden)
        self.hid_nodes_recon = np.zeros(num_hidden)
        self.likelihood = 0
        self.all_hidden_recon = np.zeros((num_visible, num_hidden))

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
        p_h = hf.sigmoid(self.bias_hid + np.dot(np.transpose(visible_nodes), self.weights))
        return p_h

    def _probability_visible(self, hidden_nodes):
        """
        Computes the conditional probability of turning on a visible unit.
        Implemented according to formula in Hinton's Practical guide to Restricted Boltzmann machine
        The probability is also just the update of the visible state.
        :param hidden_nodes: vector containing the activations of the hidden nodes
        :return: float containing probability of visible unit (single node)
        """
        p_v = hf.sigmoid(self.bias_vis + np.dot(hidden_nodes, np.transpose(self.weights)))
        return p_v

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
        # sample_nodes = np.vectorize(self._sample_node)
        # self.hid_nodes_recon = sample_nodes(self.p_h_data)
        for i in range(self.p_h_data.shape[0]):
            self.hid_nodes_recon[i] = self._sample_node(self.p_h_data[i])

        test_f.test_sampling(self.hid_nodes_recon, self.num_hidden)

        # Data outer product
        pos_grad = np.outer(vis_nodes, self.p_h_data)
        test_f.test_outer_prod_data(pos_grad, self.weights)
        return pos_grad, self.hid_nodes_recon

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
            for i in range(p_v.shape[0]):
                self.vis_nodes_recon[i] = self._sample_node(p_v[i])

            # Reconstruct the hidden nodes from visible again
            p_h = self._probability_hidden(self.vis_nodes_recon)
            # Sample from this probability distribution
            for j in range(p_h.shape[0]):
                self.hid_nodes_recon[j] = self._sample_node(p_h[j])

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
        self.likelihood += (pos_grad - neg_grad)
        # print(self.likelihood)
        self.weights += lr * self.likelihood
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

    def train(self, input_data, split=0.8, max_iterations=100, lr=0.01, k=1):
        """
        Train the restricted Boltzmann machine (RBM)
        :param input_data: Matrix containing samples and features
        :param split: Float corresponding with the
        :param max_iterations: Integer corresponding to the maximum number of iterations over one training batch
        :param lr: Float corresponding to the learning rate of the model
        """
        log_loss = np.zeros(max_iterations+1)
        epoch = 0
        gradient = float("inf")
        likelihood_train = np.zeros(max_iterations)
        likelihood_val = np.zeros(max_iterations)
        likelihood = 0
        reconstruction_cost_train = np.zeros(max_iterations)
        reconstruction_cost_val = np.zeros(max_iterations)

        while epoch < max_iterations and gradient != 0:
            # Split up the data into training (9 parts), validation (1 part)
            split_nr = int(input_data.shape[0] * split * 0.9)
            np.random.shuffle(input_data)
            train = input_data[0:split_nr, :]
            val = input_data[split_nr:, :]

            # Initialization
            reconstruction_cost = 0
            # Train RBM
            # sum loss per example and divide by the number of samples (average loss)
            for i in range(train.shape[0]):
                # This can be further vectorized
                # Do contrastive divergence algorithm
                self._update_model_params(train[i, :], lr=lr, k=k)
                # self.all_hidden_recon[i, :] = self.hid_nodes_recon

            self.weights = np.divide(self.weights, train.shape[0])
            self.bias_vis = np.divide(self.bias_vis, train.shape[0])
            self.bias_hid = np.divide(self.bias_hid, train.shape[0])

            for m in range(train.shape[0]):
                p_h = self._probability_hidden(train[m, :])
                h = np.zeros(p_h.shape[0])
                for n in range(p_h.shape[0]):
                    h[n] = self._sample_node(p_h[n])

                reconstruction_cost += self._cost_cross_entropy(train[m, :], h)
                # likelihood = self._likelihood(train[m, :], h)
            # likelihood_train[epoch] = likelihood
            # log_loss[epoch] = np.divide(np.sum(self.likelihood), train.shape[0])
            reconstruction_cost_train[epoch] = np.average(reconstruction_cost/train.shape[0])
            print("Epoch: " + str(epoch+1) + "\n")
            # print("Reconstruction cost on training set: " + str(reconstruction_cost/train.shape[0]).format("%.5f") + "\n")
            print("Average reconstruction cost on training set: " + str(reconstruction_cost/train.shape[0]).format("%.5f") + "\n")
            reconstruction_cost = 0

            # Validate RBM
            for j in range(val.shape[0]):
                p_h_val = self._probability_hidden(val[j, :])
                h_val = np.zeros(p_h_val.shape[0])
                for k in range(p_h_val.shape[0]):
                    h_val[k] = self._sample_node(p_h_val[k])

                reconstruction_cost += self._cost_cross_entropy(val[j, :], h_val)
                # likelihood_val[epoch] += self._likelihood(val[j, :], h_val)
            print("Reconstruction cost on validation set: " + str(reconstruction_cost/val.shape[0]).format("%.5f") + "\n")
            reconstruction_cost_val[epoch] = np.average(reconstruction_cost/val.shape[0])
            epoch += 1
        # Plot the likelihood
        # vis.log_likelihood_plots(range(max_iterations), likelihood_train)
        # vis.log_likelihood_plots(range(max_iterations), likelihood_val)
        plt.figure()
        plt.plot(range(max_iterations), reconstruction_cost_train)
        plt.plot(range(max_iterations), reconstruction_cost_val)
        plt.show()

    def _cost_cross_entropy(self, data, hidden_recon):
        """
        Compute cross-entropy to keep track of training of the model.
        :param data: numpy array containing the input data
        :return: float containing the reconstruction cost
        """
        y_n = hf.sigmoid(self.bias_vis + np.dot(hidden_recon, np.transpose(self.weights)))
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

    def _energy(self, vis_nodes, hid_nodes):
        """
        Function that computes the energy function using the hidden and visible nodes
        :param vis_nodes: numpy array corresponding to the visible nodes
        :param hid_nodes: numpy array corresponding to the hidden nodes
        :return:
        """
        return -np.dot(np.transpose(hid_nodes), self.bias_hid) - np.dot(np.transpose(hid_nodes), np.transpose(self.weights),
                                                                           vis_nodes) - np.dot(
            np.transpose(self.bias_vis), vis_nodes)

    def _partition_function(self, vis, hid):
        """
        :param vis_nodes: Vector corresponding with the visible nodes in the machine
        :param hid_nodes: Vector corresponding with the hidden nodes in the machine
        :return:Float corresponding to the partition function
        """
        return np.sum(np.exp(-self._energy(vis, hid)))

    def _likelihood(self, visible_nodes, hidden_nodes):
        """
        Compute the likelihood according to paper by Hinton.
        :return: Float corresponding to the likelihood of the model
        """
        nodes_visible = visible_nodes
        nodes_hidden = hidden_nodes
        vis = np.array([[1,0,1,0,0,0]])
        hid = np.transpose(np.array([[1,0,0,0]]))
        partition_function = np.sum(np.exp(-np.dot(np.transpose(hid), self.bias_hid) - np.dot(np.transpose(hid), np.transpose(self.weights),
                                                                           vis) - np.dot(
            np.transpose(self.bias_vis), vis)))
        energy = -np.dot(np.transpose(hid), self.bias_hid) - np.dot(np.transpose(hid), np.transpose(self.weights),
                                                                           vis) - np.dot(
            np.transpose(self.bias_vis), vis)
        likelihood = (1.0/partition_function) * energy
        print(nodes_visible)
        print(nodes_hidden)
        return np.average(likelihood)
