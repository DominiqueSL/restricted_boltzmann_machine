import tensorflow as tf
import help_functions as hf
import numpy as np


class RBM:
    def __init__(self, num_visible, num_hidden, depth=1):
        """
        Constructor of the restricted Boltzmann machine
        Takes care of the initialization of many variables
        depth_network: Number of layers deep the restricted Boltzmann machine is
        """
        self.num_visible = num_visible
        # Initialization
        self.hidden_nodes = np.ones(num_hidden)
        self.depth_network = depth
        self.weights = np.random.normal(0.0, 0.01, (num_visible, num_hidden))
        self.bias_hid = np.zeros(num_hidden)
        self.bias_vis = self._bias_visible(num_visible)

    # def energy(self, visible_nodes, hidden_nodes):
    #     """
    #     Compute the energy of the model given the hidden and visible layers
    #     :return: float corresponding to the energy
    #     """
    #     return - np.dot(self.bias_vis, visible_nodes) - np.dot(self.bias_hid, hidden_nodes) - \
    #            np.transpose(visible_nodes) @ self.weights @ self.hidden_nodes
    #
    # def free_energy(self):
    #     """
    #     Compute the free energy, which is required to compute the gradient of the other parameters.
    #     :return:
    #     """



    def _bias_visible(self, visible_units):
        """
        Compute the bias of the visible units
        Implementing this function could lead to infinite biases if all observations are 1
        :return: tensor containing the bias of the visible units
        """
        proportion = np.divide(np.sum(visible_units, 1), visible_units.shape[1])
        denominator = np.subtract(np.ones(proportion.shape), proportion)
        return np.log(np.divide(proportion, denominator))

    def probability_hidden_node(self, visible_nodes):
        """
        Computes the probability of turning on a hidden unit. This only applies to a single node.
        :return: Float with the probability of hidden node
        """
        return hf.sigmoid(self.bias_hid + np.dot(np.transpose(self.weights), visible_nodes))

    def probability_visible(self, hidden_nodes):
        """
        Computes the conditional probability of turning on a visible unit. This is only applicable to a single node
        Implemented according to formula in Hinton's Practical guide to Restricted Boltzmann machine
        The probability is also just the update of the visible state.
        :return: float containing probability of visible unit (single node)
        """
        return hf.sigmoid(self.bias_vis + np.dot(np.transpose(self.weights), hidden_nodes))

    # def joint_probability(self, visible_nodes, hidden_nodes):
    #     """
    #     Computes the probability of the possible combinations between hidden and visible vectors
    #     :return: float corresponding to the probability
    #     """
    #     e = self.energy(visible_nodes, hidden_nodes)
    #     z = hf.partition_function(e)
    #     return 1 / z * tf.reduce_sum(tf.exp(-e))

    def pos_gradient(self, vis_nodes):
        """
        Compute the positive gradient step, required for the contrastive divergence algorithm.
        :param vis_nodes: vector containing the activation of the visible nodes
        :return: Vector with on each index the outer product between hidden and visible node
        :return: Vector with the hidden node activation
        """
        # Positive gradient: Visible => Hidden
        apply_all_nodes = np.vectorize(self.probability_hidden_node())
        p_h = apply_all_nodes(vis_nodes)  # hid_nodes is a vector of hidden probabilities

        # Sample from this probability distribution
        sample_nodes = np.vectorize(self.update_hidden_state())
        hid_nodes_act = sample_nodes(p_h)

        # data outer product
        return np.outer(vis_nodes, hid_nodes_act), hid_nodes_act

    def neg_gradient(self, hid_nodes):
        """
        Calculate the outer product between hidden and visible nodes, which are obtained from reconstructions.
        :param hid_nodes: activations of hidden nodes
        :return: Vector with on each index the outer product between hidden and visible nodes
        :return: Vector with all the visible node activations
        :return: Vector with all the hidden node activations
        """
        # Negative gradient: Hidden=>Visible (reconstruction of data)
        apply_all_hid_nodes = np.vectorize(self.probability_visible())
        p_v = apply_all_hid_nodes(hid_nodes)

        # Sample from this probability distribution
        sample_vis = np.vectorize(self.sample_node())
        vis_nodes_act = sample_vis(p_v)
        # Reconstruct the hidden nodes from visible again
        apply_all_vis_nodes = np.vectorize(self.probability_hidden_node())
        p_h = apply_all_vis_nodes(vis_nodes_act)

        # Sample from this probability distribution
        sample_nodes = np.vectorize(self.update_hidden_state())
        hid_nodes_act = sample_nodes(p_h)

        return np.outer(vis_nodes_act, hid_nodes_act), vis_nodes_act, hid_nodes_act

    def update_model_params(self, vis_nodes, lr=0.01, k=1):
        """
        Approximate the gradient using the contrastive divergence algorithm. This is required to compute the weight update,
        as well as other model parameter updates.
        :param k: int corresponding to the number of iterations of CD
        :param lr: float corresponding to the learning rate of the model
        """
        # Compute positive gradient
        pos_grad, hid_node_act_data = self.pos_gradient(vis_nodes)

        # Iterate k number of times
        for i in range(k):
            neg_grad, vis_nodes_act, hid_nodes_act = self.neg_gradient(hid_node_act_data)
            self.weights += lr*(pos_grad-neg_grad)
            self.bias_hid += lr*(hid_node_act_data - hid_nodes_act)
            self.bias_vis += lr*(vis_nodes - vis_nodes_act)

    def update_hidden_state(self, prob):
        """
        Node is switched on if the input probability is higher than number random uniformly sampled between 0 and 1
        :param prob: float representing the probability
        :return: 0 or 1, respectively corresponding to not active and active.
        This function only required if we do more steps of contrastive divergence
        """
        if prob > np.random.uniform(0, 1):
            return 1
        else:
            return 0

    def sample_node(self, prob):
        """
        Obtain unbiased sample of node. Sample from conditional probability. Needed to perform Gibbs sampling step.
        :return: binary number corresponding with the activation of the node
        """
        return np.random.binomial(1, prob) # if you sample binomial once you sample bernoulli

    # def sample_h_from_v(self, prob):
    #     """
    #     Obtain unbiased sample of h. Sample from conditional probability p(h|v). Required to perform the Gibbs sampling
    #     step.
    #     :prob: Tensor containing the probability of returning a 1
    #     :return: float corresponding with probability
    #     """
    #     # We need the probability of having an activation event
    #     # This is input for tf.Bernoulli distribution
    #     return np.random.binomial(1, prob) # if you sample binomial once you sample bernoulli


    # def gibbs_sampling_vhv(self, prob):
    #     """
    #     Perform one iteration of Gibbs sampling starting from the visible units
    #     :return: Binary state of the visible unit
    #     """
    #     sample_h = self.sample_h_from_v(prob)
    #     prob_h = self.prob
    #     return self.sample_v_from_h(sample_h)

    def reconstruction_error(self, visible_nodes, recon_visible_nodes):
        """
        Compute the reconstruction error between the original data and the predictions, using the cross-entropy
        Hinton's practical guide to RBM noted that this is most appropriate for the restrictive boltzmann machine
        Easy way out is to just compute the root mean squared error
        :param visible_nodes: Vector containing the original data
        :param recon_visible
        :return: error
        """
        # Simply take the squared error between the original input data and the reconstructed data
        norm = np.linalg.norm(visible_nodes - recon_visible_nodes)
        return np.square(norm)

    # def gibbs_sampling_hvh(self, prob):
    #     """
    #     Perform one iteration of Gibbs sampling starting from the hidden units
    #     :return: Binary state of the hidden node
    #     """
    #     sample_v = self.sample_v_from_h(prob)
    #     return self.sample_h_from_v(sample_v)

    def train(self, input_data, max_epochs=100, lr=0.01, error_threshold=0.1, batchsize=10):
        """
        Train the restricted Boltzmann machine (RBM)
        :input_data: Tensor containing all the
        """
        epoch = 0
        # Still need to include the division into batches!
        while epoch <= max_epochs or error > error_threshold:
            for batch in input_data.shape[1]/batchsize:

                # Compute error
                error = self.reconstruction_error(visible_nodes, recon_visible_nodes)
                print("Error: %.3f".format(error) + "\n")
                print("Epoch: " + str(epoch))

                # Updates
                self.update_weight(lr)
                # Update biases

            epoch += 1


if __name__ == '__main__':
    training_data = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1],
         [0, 0, 1]])
    r = RBM(num_visible=training_data.shape[0], num_hidden=3)
    test_prob = r.probability_hidden(training_data[:, 1])
    print(r.gibbs_sampling_hvh(test_prob[1]))
