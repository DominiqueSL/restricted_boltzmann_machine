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
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.input_data = training_data
        self.weights = np.random.normal(0.0, 0.01, (self.num_visible, self.num_hidden))
        self.bias_hid = np.zeros(self.num_hidden)
        self.bias_vis = self._bias_visible_init(training_data)
        # self.bias_vis = np.zeros(self.num_visible) # in case everything is 1 or zero

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

    @staticmethod
    def _sample_nodes(prob):
        """
        Obtain unbiased sample of node. Sample from conditional probability. Needed to perform Gibbs sampling step.
        :param prob: numpy array corresponding with the probability of the nodes being activated
        :return: binary number corresponding with the activation of the node
        """
        samples = np.random.binomial(1, prob)  # If you sample binomial once you sample Bernoulli
        # test_f.test_sampling(samples, prob.shape[0])
        return samples

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
        self.hid_nodes_recon = self._sample_nodes(self.p_h_data)
        # test_f.test_sampling(self.hid_nodes_recon, self.num_hidden)
        # Data outer product
        pos_grad = np.outer(vis_nodes, self.p_h_data)
        # Fix this double assignment later
        # test_f.test_outer_prod_data(pos_grad, self.weights)
        return pos_grad, self.hid_nodes_recon

    def _neg_gradient(self, hid_recon, k=1):
        """
        Compute the outer product between hidden and visible nodes, which are obtained from reconstructions.
        :param k: Integer corresponding to the number of Gibbs sampling steps
        :return: Vector with on each index the outer product between hidden and visible nodes
        :return: Vector with all the visible node activations
        :return: Vector with all the hidden node activations
        """
        hid_recon = np.array(hid_recon) # Check if removing this still produces the same output

        # Perform Gibbs sampling
        for step in range(k):
            # Negative gradient: Hidden => Visible (reconstruction of data)
            p_v = self._probability_visible(hid_recon)
            # Sample
            self.vis_nodes_recon = self._sample_nodes(p_v)
            # test_f.test_sampling(self.vis_nodes_recon, self.num_visible)

            # Reconstruct the hidden nodes from visible again
            p_h = self._probability_hidden(self.vis_nodes_recon)
            # Sample from this probability distribution
            # test_f.test_sampling(self.hid_nodes_recon, self.num_hidden)
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

        # Use k number of steps Gibbs sampling
        neg_grad, p_v_recon, p_h_recon = self._neg_gradient(hid_node_recon_data, k=k)

        # Compute reconstruction error
        recon = hf.cross_entropy(vis_nodes, p_v_recon)

        # Update
        self.gradient_update += (lr * (pos_grad - neg_grad)) / train_size
        self.hid_update += (lr * (self.p_h_data - p_h_recon)) / train_size
        self.vis_update += (lr * (vis_nodes - p_v_recon)) / train_size
        return recon

    def _recon_training(self, training_set, lr=0.01, k=1):
        """
        Update the model parameters, by using function to compute the model parameters, and dividing by the number of
        samples.
        :param training_set: numpy array containing the training samples
        :param lr: float corresponding to the learning rate of the model
        :param k: integer corresponding to the number of Contrastive Divergence steps
        """
        recon = np.zeros(self.num_visible) # Initialize
        recon_rmse = 0
        # Loop over all the training vectors
        for i in range(training_set.shape[0]):
            # This can possibly be further vectorized
            # Do contrastive divergence algorithm
            recon += self._compute_model_params(training_set[i, :], training_set.shape[0], lr=lr, k=k)
            recon_rmse += hf.sum_squared_recon_error(training_set[i, :], self.vis_nodes_recon)
            self.all_p_h_data[i, :] = self.p_h_data # Needed for the probability images
        return np.average(recon / training_set.shape[0]), recon_rmse / training_set.shape[0]

    def make_prediction(self, sample_data):
        """
        Makes prediction of the state of the visible nodes
        :param sample_data: Sample of the class you want to predict. Predict hidden nodes
        :return: Nothing
        """
        # Calculate the hidden probability
        p_hid = self._probability_hidden(sample_data)
        # Sample nodes
        h = self._sample_nodes(p_hid)
        # Calculate the visible activation probability
        p_vis = self._probability_visible(h)
        # Sample back to visible again
        v = self._sample_nodes(p_vis)
        return p_vis, v
        # print(h)
        # print(v)

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
        sum_recon_cost = 0
        # Loop over all samples again, with the learnt weights and biases
        for m in range(data_set.shape[0]): # Possibly vectorize so we can omit this for-loop
            p_v, v = self.make_prediction(data_set[m, :])
            # Compute the reconstruction cost, sum over all the training samples
            reconstruction_cost += hf.cross_entropy(data_set[m, :], p_v)
            sum_recon_cost += hf.sum_squared_recon_error(data_set[m, :], v)
            # test_f.test_cross_entropy(reconstruction_cost, p_v.shape[0])
        # Average out the reconstruction cost, by averaging it over all the nodes etc.
        reconstruction_cost_tot = np.average(reconstruction_cost / data_set.shape[0])
        sum_recon_cost_tot = sum_recon_cost / data_set.shape[0]
        return reconstruction_cost_tot, sum_recon_cost_tot

    def _update_parameters(self):
        """
        Update all the model parameters (weights, biases)
        """
        # Normalize all the parameters by dividing by the number of training samples
        # Check later if division goes well (is every index divided by scalar)
        self.weights += self.gradient_update
        self.bias_vis += self.vis_update
        self.bias_hid += self.hid_update

    def _initialize_model_param(self, train_size):
        """
        Initialize the model parameters to zero
        :param train_size: integer corresponding to the size of the training data set
        """
        self.gradient_update = 0
        self.hid_update = 0
        self.vis_update = 0
        self.all_p_h_data = np.zeros((train_size, self.num_hidden))

    def train(self, batchsize, split=0.8, max_iterations=100, lr=0.01, k=1):
        """
        Train the restricted Boltzmann machine (RBM)
        :param batchsize: Integer corresponding with the batch size
                          Use batchsize training to speed up algorithm
        :param split: Float corresponding with the split into train and testing set
        :param max_iterations: Integer corresponding to the maximum number of iterations over one training batch
        :param lr: Float corresponding to the learning rate of the model
        :param k: Integer corresponding with the number of contrastive divergence iterations
        """
        reconstruction_cost_train = np.zeros(max_iterations)
        reconstruction_cost_val = np.zeros(max_iterations)
        rmse_train = np.zeros(max_iterations)
        rmse_val = np.zeros(max_iterations)
        # Split into train and test
        train_set = self.input_data[:int(split * self.input_data.shape[0]), :]
        # Iterate over training set to train the RBM until conditions are met
        for epoch in range(max_iterations):
            # Split up the data into training (9 parts), validation (1 part)
            train_split = int(train_set.shape[0] * 0.9)
            # Shuffle data set
            np.random.shuffle(train_set)
            train = train_set[:train_split, :]
            val = train_set[train_split:, :]
            # Initialize model parameters
            self._initialize_model_param(train.shape[0])
            # Perform batch-wise training
            # Update parameters and compute reconstructions
            for batch in range(batchsize):
                reconstruction_cost_train[epoch], rmse_train[epoch] = self._recon_training(train[batch, :], lr=lr, k=k)
                # Update parameters
                self._update_parameters()
            # Visualize the hidden probability activation and directly save output.
            if epoch % 100 == 0:
                vis.visualize_hidden_prob_activation(self.all_p_h_data, "Hidden_probability_activation_brain_data_" + str(epoch))
                vis.model_param_visualization(self.weights, self.bias_vis, self.bias_hid, self.gradient_update, self.vis_update, self.hid_update, "parameter_histogram_brain_data_" + str(epoch))

            # Print output
            print("Epoch: " + str(epoch+1) + "\n")
            print("RMSE (train): " + str(rmse_train[epoch]))
            print("Average cross-entropy (train): " + str(reconstruction_cost_train[epoch]).format("%.5g") + "\n")
            reconstruction_cost_val[epoch], rmse_val[epoch] = self._compute_reconstruction_cost_val(val)
            print("Average cross-entropy (validation): " + str(reconstruction_cost_val[epoch]).format("%.5g") + "\n")
            print("RMSE (validation): " + str(rmse_val[epoch]) + " \n")

        # print(self.all_p_h_data)
        # print(self.weights)
        # Plot the loss
        vis.loss_plots(range(max_iterations), reconstruction_cost_train, reconstruction_cost_val, "loss_plot_brain_")
        vis.loss_plots(range(max_iterations), rmse_train, rmse_val, " loss_plot_rmse_brain_")
        # Save weights and biases
        hf.write_h5py(self.weights, filename="final_weights.h5")
        hf.write_h5py(self.bias_hid, filename="final_vis_bias.h5")
        hf.write_h5py(self.bias_vis, filename="final_hid_bias.h5")

    def test(self, data_array, split):
        """
        Function that loops over test set and computes the reconstruction on the test set
        :param data_array: numpy array corresponding to the training data
        :param split: float corresponding to the split of the data set into train and test
        """
