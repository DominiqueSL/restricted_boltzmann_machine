# Import required libraries
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import visualization as vis
import rbm
import help_functions as hf
import pandas as pd

# Main program to train Restricted Boltzmann Machine
# training_data = np.array(
#     [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
#      [0, 0, 1, 1, 1, 0]])
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# image_data = np.transpose(hf.read_data("./boltzmann_machine_toy_data.csv"))
image_data = pd.DataFrame.as_matrix(pd.read_csv("./boltzmann_machine_toy_data.csv"))
num_hidden = 2
# train_images = np.transpose(np.where(mnist.train._images[0]>0))
# mnist.train._images[mnist.train._images[[0]] >=0.5] = 1
# mnist.train._images[[0]<0.5] = 0

r = rbm.RBM(training_data=image_data, num_visible=image_data.shape[1], num_hidden=num_hidden)
r.train(image_data, max_epochs=5000, lr=0.1)

# Save output


# visualization