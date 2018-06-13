# Import required libraries
import rbm
import help_functions as hf
# import pandas as pd
import sys

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
# training_data = pd.DataFrame.as_matrix(pd.read_csv("./binary/20080516_R2_1.csv", header=None))


def main(dirname, num_hidden):
    """
    Main program for running the restricted Boltzmann machine
    :param dirname: string with name of the directory in which the input files are present
    :param num_hidden: integer corresponding with the number of hidden nodes
    """
    data_split = 0.8

    # Select a range of number of hidden nodes, which needs to be optimized
    # num_hidden = np.array(range(100, 50, 200))

    # filepath = "./binary/"
    filepath = "./"

    # training_data = hf.load_hdf5(filepath + dirname + "/brain_data_set.hdf5")
    path = r"./MNIST_data/train-images-idx3-ubyte.gz"
    train_set_images = hf.extract_data(path, 60000)
    print("Finished loading data. Start training")
    # print(train_set_images.shape)
    r = rbm.RBM(training_data=train_set_images, num_visible=train_set_images.shape[1], num_hidden=int(num_hidden))

    # r = rbm.RBM(training_data=training_data, num_visible=training_data.shape[1], num_hidden=int(num_hidden))
    r.train(outfile=dirname, split=data_split, max_iterations=1000, lr=0.1, k=1)
    # r.test(split=data_split)
    # r.make_prediction(np.transpose(np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])))
    # r.make_prediction(np.transpose(np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])))

    # Save output


if __name__ == '__main__':
    # script = sys.argv[0]
    # filename = sys.argv[1]
    # num_hidden = sys.argv[2]
    # main(filename, num_hidden)
    main("mnist", 100) # Local run
