import h5py
import glob
import pandas as pd
import os
import numpy as np

# Import large amount of data and write it to an h5py file for easy loading to the model


def convert_to_h5py(filepath, folder, filename, heading=None, delimiter=","):
    """
    Convert files in the given file path to a full h5py file.
    :param filepath: string with path to the files to be imported and converted to h5py
    :param filename: string with the name of the output file
    :param heading: boolean corresponding with if there is a header in the file or not
    :param delimiter: string corresponding with the delimiter in the file to be converted
    """
    # Function still needs to be tested
    # Read file
    if delimiter == ",":
        # First we need to find out the number of features, because we need to know how large the dataset should be
        # column wise
        first_file = os.listdir(filepath)[0]
        file = pd.DataFrame(pd.read_csv(filepath + first_file, delimiter=",", header=heading))
        nr_cols = len(file.columns)

        # Initialize the dataset
        with h5py.File(filepath + filename + ".hdf5", 'a') as hf:
            # Start reading all the files
            names = glob.glob(filepath + folder + "/" + "*.csv")
            # Append the dataset together using h5py
            for index, name in enumerate(names, start=1):
                if index == 1:
                    data = pd.DataFrame.as_matrix(pd.read_csv(name, header=heading)) # training matrix
                else:
                    data = np.concatenate((data, pd.DataFrame.as_matrix(pd.read_csv(name, header=heading)))) # training matrix
            hf.create_dataset("dataset", data=data, dtype=data.dtype)

    else:
        raise NotImplementedError("Other filetypes are not yet supported")


filepath = "./binary/"
for _, dirnames, _ in os.walk(filepath):
    for index, folder in enumerate(dirnames, start=1):
        convert_to_h5py(filepath, folder,  "brain_data_set_" + str(index))

print("Success")
