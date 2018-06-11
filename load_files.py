import h5py
import glob
import pandas as pd
import os

# Import large amount of data and write it to an h5py file for easy loading to the model


def convert_to_h5py(filepath, heading=None, delimiter=","):
    """
    Convert files in the given file path to a full h5py file.
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
        dset = h5py.File("brain_data_set", "x")
        fset = dset.create_dataset("ints", (100,), dtype='i8', maxshape=(None, nr_cols))
        with h5py.File(fset, 'a') as hf:
            # Start reading all the files
            names = glob.glob("*.csv")
            # Append the dataset together using h5py
            for name in names:
                data = pd.DataFrame.as_matrix(pd.read_csv(filepath + name, header=heading)) # training matrix
                hf["dataset"].resize((hf["dataset"].shape[0] + data.shape[0]), axis=0)
                hf["dataset"][-data.shape[0]:] = data
        return dset
    else:
        raise NotImplementedError("Other filetypes are not yet supported")


filpath = "./binary/"
convert_to_h5py(filepath=filpath)
print("Success :D")
