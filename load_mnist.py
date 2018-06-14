import help_functions as hf

path = r"./MNIST_data/train-images-idx3-ubyte.gz"
train_set_images = hf.extract_data(path, 60000)
hf.write_h5py(train_set_images, "mnist_compress")
print("Success")