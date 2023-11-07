import h5py
import numpy as np
from keras.utils import to_categorical

from network import NN
from dense import Dense
from convolution import Convolution
from sigmoid import Sigmoid
from reshape import Reshape
from losses import mse, mse_prime

# Constants
NUM_CLASSES = 3
NUM_NODES = 200

# Function to preprocess data
def preprocess_data(x, y, limit):
    indices = []
    # Selecting indices for each class
    for i in range(NUM_CLASSES):
        # pick digits from 0 to NUM_CLASSES - 1
        class_indices = np.where(y == i)[0][:limit]
        indices.append(class_indices)
    # Combining indices and shuffling
    all_indices = np.hstack(indices)
    all_indices = np.random.permutation(all_indices)
    # Selecting and reshaping data
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    y = to_categorical(y)
    y = y.reshape(len(y), NUM_CLASSES, 1)
    return x, y

# Loading MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))

# Reshaping data
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
MNIST_data.close()

# Preprocessing data
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# Defining network layers
network_layers = [
    Convolution((1, 28, 28), 3, 5),  # Convolution layer
    Sigmoid(),  # Activation layer
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),  # Reshape layer
    Dense(5 * 26 * 26, NUM_NODES),  # Dense layer
    Sigmoid(),  # Activation layer
    Dense(NUM_NODES, NUM_CLASSES),  # Dense layer
    Sigmoid()  # Activation layer
]

# Creating network
CNN = NN(network_layers)

# Training network
CNN.train(
    mse,
    mse_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# Initializing counters
correct_predictions = 0
total_predictions = 0

# Testing network
for x, y in zip(x_test, y_test):
    prediction = CNN.predict(x)
    if np.argmax(prediction) == np.argmax(y):
        correct_predictions += 1
    total_predictions += 1

# Printing accuracy
print(f"Accuracy: {correct_predictions / total_predictions * 100}%")
