import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def load_planar_dataset(seed):
    np.random.seed(seed)

    m = 400
    N = int(m/2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4

    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.randn(N)*0.2
        r = a*np.sin(4*t) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    return X.T, Y.T

def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache

def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1./m) * np.dot(dZ3, A2.T)
    db3 = (1./m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * (A2 > 0)
    dW2 = (1./m) * np.dot(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (A1 > 0)
    dW1 = (1./m) * np.dot(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dZ3": dZ3, "dW3": dW3, "db3": db3,
            "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
            "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}


def update_parameters(parameters, grads, learning_rate):
    n = len(parameters) // 2

    for k in range(n):
        parameters["W" + str(k+1)] -= learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] -= learning_rate * grads["db" + str(k+1)]

    return parameters



def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m), dtype=int)

    a3, _ = forward_propagation(X, parameters)

    p = (a3 > 0.5).astype(int)

    print("Accuracy:", np.mean(p == y))
    return p


def compute_cost(a3, Y):
    m = Y.shape[1]
    epsilon = 1e-8
    cost = (-1/m) * np.sum(Y * np.log(a3 + epsilon) + (1 - Y) * np.log(1 - a3 + epsilon))
    return np.squeeze(cost)


def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_orig / 255
    test_set_x = test_set_x_orig / 255

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


def predict_dec(parameters, X):
    a3, _ = forward_propagation(X, parameters)
    return (a3 > 0.5)


def load_planar_dataset_v2(randomness, seed):
    np.random.seed(seed)

    m = 50
    N = int(m/2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')

    for j in range(2):
        ix = range(N*j, N*(j+1))

        if j == 0:
            t = np.linspace(j, 4*3.1415*(j+1), N)
            r = 0.3*np.square(t) + np.random.randn(N)*randomness
        else:
            t = np.linspace(j, 2*3.1415*(j+1), N)
            r = 0.2*np.square(t) + np.random.randn(N)*randomness

        X[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
        Y[ix] = j

    return X.T, Y.T


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min()-1, X[0, :].max()+1
    y_min, y_max = X[1, :].min()-1, X[1, :].max()+1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def load_2D_dataset():
    data = scipy.io.loadmat('data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)
    return train_X, train_Y, test_X, test_Y