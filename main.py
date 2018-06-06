# Import required libraries
import rbm
import pandas as pd

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
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
image_data = pd.DataFrame.as_matrix(pd.read_csv("./boltzmann_machine_toy_data.csv"))
# training_data = pd.DataFrame.as_matrix(pd.read_csv("./binary/20080516_R1_1.csv"))
num_hidden = 500
# train_images = np.transpose(np.where(mnist.train._images[0]>0))

# r = rbm.RBM(training_data=training_data[:round(training_data.shape[0]*0.1), :], num_visible=training_data.shape[1], num_hidden=num_hidden)
r = rbm.RBM(training_data=image_data, num_visible=image_data.shape[1], num_hidden=num_hidden)
r.train(split=0.8, max_iterations=5000, lr=0.05, k=1)
# r.make_prediction(np.transpose(np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])))
# r.make_prediction(np.transpose(np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])))

# Save output


