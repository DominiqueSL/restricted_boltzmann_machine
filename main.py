# Import required libraries
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import visualization as vis
import rbm
import help_functions as hf
import pandas as pd

# Main program to train Restricted Boltzmann Machine
# training_data = np.array(
#         [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0],
#          [0, 0, 1, 1, 1, 0]])
training_data = np.array(
    [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
     [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
     [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
image_data = pd.DataFrame.as_matrix(pd.read_csv("./boltzmann_machine_toy_data.csv"))
# training_data = pd.DataFrame.as_matrix(pd.read_csv("./binary/20080516_R1_1.csv"))
num_hidden = 3
# train_images = np.transpose(np.where(mnist.train._images[0]>0))
# mnist.train._images[mnist.train._images[[0]] >=0.5] = 1
# mnist.train._images[[0]<0.5] = 0

np.random.seed(0)
r = rbm.RBM(training_data=image_data, num_visible=image_data.shape[1], num_hidden=num_hidden)
r.train(split=0.8, max_iterations=1000, lr=0.001, k=1)
print(r.weights)
# r.make_prediction(np.transpose(np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])))
# r.make_prediction(np.transpose(np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])))

# Save output


# visualization
