# Import required libraries
import rbm
import pandas as pd
import numpy as np

# Main program to train Restricted Boltzmann Machine
# training_data = np.array(
#         [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0],
#          [0, 0, 1, 1, 1, 0]])
# training_data = np.array(
#     [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#      [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#      [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#      [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
#      [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
#      [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
# )
training_data = pd.DataFrame.as_matrix(pd.read_csv("./binary/20080516_R2_1.csv", header=None))
num_hidden = 500
data_split = 0.8

# Select a range of number of hidden nodes, which needs to be optimized
# num_hidden = np.array(range(100, 50, 200))

r = rbm.RBM(training_data=training_data, num_visible=training_data.shape[1], num_hidden=num_hidden)
# r = rbm.RBM(training_data=image_data, num_visible=image_data.shape[1], num_hidden=num_hidden)
r.train(split=data_split, max_iterations=5000, lr=0.1, k=1)
r.test(split=data_split)
# r.make_prediction(np.transpose(np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])))
# r.make_prediction(np.transpose(np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])))

# Save output


