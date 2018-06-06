import numpy as np
import help_functions as hf
import test_individual_function as test_f
import visualization as vis


class RBM:
    def __init__(self, training_data, num_visible, num_hidden):
        """
        Constructor of the restricted Boltzmann machine
        Takes care of the initialization of many variables
        :param training_data: Matrix containing all the data required for training of the network
        :param num_visible: Integer corresponding to the number of visible units in the network
        :param num_hidden: Integer corresponding to the number of hidden units in the network
        """
        training_data = np.array(training_data, copy=True) # possibly delete later
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.input_data = training_data
        self.hidden_nodes = np.ones(num_hidden)
        # self.weights = np.random.normal(0.0, 0.01, (self.num_visible, self.num_hidden))
        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))
        self.bias_hid = np.zeros(self.num_hidden)
        # self.bias_vis = self._bias_visible_init(training_data)
        self.bias_vis = np.zeros(self.num_visible) # in case everything is 1 or zero
        self.vis_update = np.zeros(self.num_visible)
        self.hid_update = np.zeros(self.num_hidden)
        self.vis_nodes_recon = np.zeros(self.num_visible)
        self.p_h_data = np.zeros(self.num_hidden)
        self.hid_nodes_recon = np.zeros(self.num_hidden)
        self.gradient = float("inf") # Maybe delete later
        self.gradient_update = np.zeros((self.num_visible, self.num_hidden))

        # Ugly inclusions for debugging
        self.all_p_h_data = 0
        self.recon_error = 0

    @staticmethod
    def _bias_visible_init(visible_units):
        """
        Compute the initial bias of the visible units
        :param visible_units: The activations of the visible units
        :return: tensor containing the bias of the visible units
        """
        visible_units = np.array(visible_units, copy=True)
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

    @staticmethod
    def _sample_node(prob):
        """
        Obtain unbiased sample of node. Sample from conditional probability. Needed to perform Gibbs sampling step.
        :param prob: float corresponding with the probability of a node being activated
        :return: binary number corresponding with the activation of the node
        """
        return np.random.binomial(1, prob)  # If you sample binomial once you sample Bernoulli

    def _pos_gradient(self, vis_nodes):
        """
        Compute the positive gradient step, required for the contrastive divergence algorithm.
        :param vis_nodes: Vector containing the activation of the visible nodes
        :return: Vector with on each index the outer product between hidden and visible node
        :return: Vector with the hidden node activation
        """
        # Positive gradient: Visible => Hidden
        p_h_data = self._probability_hidden(vis_nodes)
        # Sample from this probability distribution
        hid_nodes_recon = np.zeros(p_h_data.shape[0])
        for i in range(p_h_data.shape[0]):
            hid_nodes_recon[i] = self._sample_node(p_h_data[i])

        # test_f.test_sampling(self.hid_nodes_recon, self.num_hidden)
        # Data outer product
        pos_grad = np.outer(vis_nodes, p_h_data)
        self.p_h_data = np.array(p_h_data)
        self.hid_nodes_recon = np.array(hid_nodes_recon)
        # test_f.test_outer_prod_data(pos_grad, self.weights)
        return pos_grad, hid_nodes_recon

    def _neg_gradient(self, hid_recon, k=1):
        """
        Compute the outer product between hidden and visible nodes, which are obtained from reconstructions.
        :param k: Integer corresponding to the number of Gibbs sampling steps
        :return: Vector with on each index the outer product between hidden and visible nodes
        :return: Vector with all the visible node activations
        :return: Vector with all the hidden node activations
        """
        hid_recon = np.array(hid_recon)

        # Perform Gibbs sampling
        for step in range(k):
            # Negative gradient: Hidden => Visible (reconstruction of data)
            p_v = self._probability_visible(hid_recon)
            # Sample
            vis_nodes_recon = np.zeros(p_v.shape[0])
            for i in range(p_v.shape[0]):
                vis_nodes_recon[i] = self._sample_node(p_v[i])
            # test_f.test_sampling(self.vis_nodes_recon, self.num_visible)

            # Reconstruct the hidden nodes from visible again
            p_h = self._probability_hidden(vis_nodes_recon)
            # Sample from this probability distribution
            for j in range(p_h.shape[0]):
                hid_recon[j] = self._sample_node(p_h[j])
            # test_f.test_sampling(self.hid_nodes_recon, self.num_hidden)
            self.vis_nodes_recon = np.array(vis_nodes_recon)
        return np.outer(p_v, p_h), p_v, p_h

    def _compute_model_params(self, vis_nodes, train_size, lr=0.01, k=1):
        """
        Approximate the gradient using the contrastive divergence algorithm. This is required to compute the weight update,
        as well as other model parameter updates.
        :param vis_nodes vector containing the input data from the visible nodes
        :param lr: float corresponding to the learning rate of the model
        :param k: int corresponding to the number of iterations using CD

        :return: Reconstructed visible nodes are returned, as they are needed to compute the error
        """
        # Compute positive gradient
        vis_nodes = np.array(vis_nodes)

        pos_grad, hid_node_recon_data = self._pos_gradient(vis_nodes)

        # Iterate k number of times
        # for i in range(k):
        neg_grad, p_v_recon, p_h_recon = self._neg_gradient(hid_node_recon_data, k=k)

        # Compute reconstruction error
        recon = self._cost_cross_entropy(vis_nodes, p_v_recon)

        # Update
        # Differences here compared to the old code
        # self.gradient = np.average(np.average(lr * (pos_grad - neg_grad))) # Possibly delete
        # self.bias_vis and self.bias_hid are somehow changed here.
        self.gradient_update += lr * (pos_grad - neg_grad) * (1.0 / train_size)
        self.hid_update += lr * (self.p_h_data - p_h_recon) * (1.0 / train_size)
        self.vis_update += lr * (vis_nodes - p_v_recon) * (1.0 / train_size)
        self.recon_error += np.sum((vis_nodes - self.vis_nodes_recon)**2)

        return recon

    def _recon_training(self, training_set, lr=0.01, k=1):
        """
        Update the model parameters, by using function to compute the model parameters, and dividing by the number of
        samples.
        :param training_set: numpy array containing the training samples
        :param lr: float corresponding to the learning rate of the model
        :param k: integer corresponding to the number of Contrastive Divergence steps
        """
        training_set = np.array(training_set)
        recon = np.zeros(self.num_visible) # Initialize
        # Loop over all the training vectors
        for i in range(training_set.shape[0]):
            # This can be further vectorized
            # Do contrastive divergence algorithm
            recon += self._compute_model_params(training_set[i, :], training_set.shape[0], lr=lr, k=k)
            self.all_p_h_data[i, :] = self.p_h_data

        return np.average(recon/training_set.shape[0])

    def _compute_reconstruction_cost_val(self, data_set):
        """
        Function that computes the full reconstruction error
        Error is computed using the cross-entropy, as stated in Hinton's Practical Guide to training Boltzmann  machines
        => Most appropriate for Contrastive Divergence
        :param data_set: numpy array with the dataset on which we have to compute the error
        :return: Float corresponding to the reconstruction cost of the dataset
        """
        # Initialize the reconstruction cost
        reconstruction_cost = 0
        # Loop over all samples again, with the learnt weights and biases
        for m in range(data_set.shape[0]):
            p_h = self._probability_hidden(data_set[m, :])
            h = np.zeros(p_h.shape[0])
            for n in range(p_h.shape[0]):
                h[n] = self._sample_node(p_h[n])
            p_v = self._probability_visible(h)
            # Compute the reconstruction cost, sum over all the training samples
            reconstruction_cost += self._cost_cross_entropy(data_set[m, :], p_v)

        # Average out the reconstruction cost, by averaging it over all the nodes etc.
        reconstruction_cost_tot = np.average(reconstruction_cost / data_set.shape[0])
        # return reconstruction_cost_tot, log_likelihood, reconstruction_error
        return reconstruction_cost_tot

    def _update_parameters(self, train_size):
        """
        Update all the model parameters (weights, biases)
        """
        # Normalize all the parameters by dividing by the number of training samples
        # Check later if division goes well (is every index divided by scalar)
        self.weights += self.gradient_update
        self.bias_vis += self.vis_update
        self.bias_hid += self.hid_update

    def train(self, split=0.8, max_iterations=100, lr=0.01, k=1):
        """
        Train the restricted Boltzmann machine (RBM)
        :param split: Float corresponding with the split into train and testing set
        :param max_iterations: Integer corresponding to the maximum number of iterations over one training batch
        :param lr: Float corresponding to the learning rate of the model
        :param k: Integer corresponding with the number of contrastive divergence iterations
        """
        epoch = 0
        reconstruction_cost_train = np.zeros(max_iterations)
        reconstruction_cost_val = np.zeros(max_iterations)
        # Split into train and test
        train_set = self.input_data[:int(split * self.input_data.shape[0]), :]
        # Iterate over training set to train the RBM until conditions are met
        while epoch < max_iterations:
            # Split up the data into training (9 parts), validation (1 part)
            train_split = int(train_set.shape[0] * 0.9)
            # Shuffle dataset
            np.random.shuffle(train_set)
            train = train_set[:train_split, :]
            val = train_set[train_split:, :]
            # Initialization
            self.all_p_h_data = np.zeros((train.shape[0], self.num_hidden))

            # Perform batch-wise training
            # Incorporate batch size learning later in here
            # Update parameters and compute reconstructions
            reconstruction_cost_train[epoch] = self._recon_training(train, lr=lr, k=k)
            # Visualize the hidden probability activation and directly save output.
            if epoch % 100 == 0:
                vis.visualize_hidden_prob_activation(self.all_p_h_data, "Hidden_probability_activation_" + str(epoch))
                vis.model_param_visualization(self.weights, self.bias_vis, self.bias_hid, self.gradient_update, self.vis_update, self.hid_update, "parameter_histogram_" + str(epoch))

            print("Epoch: " + str(epoch+1) + "\n")
            print("Recon error RMSE: " + str(self.recon_error / train.shape[0]))
            print("Average reconstruction cost on training set: " + str(reconstruction_cost_train[epoch]).format("%.5f") + "\n")
            # reconstruction_cost_val[epoch] = self._compute_reconstruction_cost_val(val)
            # print("Average reconstruction cost on validation set: " + str(reconstruction_cost_val[epoch]).format("%.5f") + "\n")

            # Update takes place here
            self._update_parameters(train.shape[0])
            self.recon_error = 0
            self.gradient_update = 0
            self.hid_update = 0
            self.vis_update = 0
            epoch += 1
        # Plot the loss
        print(self.all_p_h_data)
        print(self.weights)
        vis.loss_plots(range(epoch), reconstruction_cost_train, reconstruction_cost_val)


    def test(self, test_data_set):
        """
        Run the final weights with the test set and see model performance
        :param test_data_set: numpy array corresponding with the test data set
        """

    @staticmethod
    def _cost_cross_entropy(data, y_n):
        """
        Compute cross-entropy to keep track of training of the model.
        :param data: numpy array containing the input data
        :return: float containing the reconstruction cost
        """
        cross_entropy = data * np.log(y_n) + (1 - data) * np.log(1 - y_n)
        return -cross_entropy

    def make_prediction(self, bias_hid, bias_vis, sample_data):
        """
        Makes prediction of the state of the visible nodes
        :param sample_data: Sample of the class you want to predict. Predict hidden nodes
        :return: Nothing
        """
        # Calculate the hidden probability
        p_h = self._probability_hidden(sample_data)
        h = np.zeros(p_h)

        # Sample nodes
        for node in range(p_h):
            h[node] = self._sample_node(p_h[node])

        # Calculate the visible activation probability
        p_v = self._probability_visible(h)

        v = np.zeros(p_v)
        # Sample back to visible again
        for v_node in range(p_v):
            v[v_node] = self._sample_node(p_v[v_node])
        print(h)
        print(v)