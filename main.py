# Import required libraries
import rbm
import help_functions as hf
import sys
# Main program to train Restricted Boltzmann Machine


def main(dirname, num_hidden):
    """
    Main program for running the restricted Boltzmann machine
    :param dirname: string with name of the directory in which the input files are present
    :param num_hidden: integer corresponding with the number of hidden nodes
    """
    DATA_SPLIT = 0.8

    # Select a range of number of hidden nodes, which needs to be optimized
    # num_hidden = np.array(range(100, 50, 200))

    filepath = "./binary/"
    # filepath = "./"
    training_data = hf.load_hdf5(filepath + dirname + "/brain_data_set.hdf5")
    # train_set_images = hf.load_hdf5(filepath + dirname + ".hdf5")
    print("Finished loading data. Start training")
    r = rbm.RBM(training_data=training_data, num_visible=training_data.shape[1], num_hidden=int(num_hidden))

    # r = rbm.RBM(training_data=training_data, num_visible=training_data.shape[1], num_hidden=int(num_hidden))
    r.train(outfile=dirname, split=DATA_SPLIT, max_iterations=1000, lr=0.1, k=1, visualize=False)
    r.test(split=DATA_SPLIT)
    r.save_parameters(dirname)  # Save output


if __name__ == '__main__':
    # script = sys.argv[0]
    # filename = sys.argv[1]
    # num_hidden = sys.argv[2]
    # main(filename, num_hidden)
    main("", 100) # Local run
