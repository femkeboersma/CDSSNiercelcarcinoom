import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def one_hot_encoding(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def calculate_accuracy(Y_true, Y_pred):
    correct_predictions = np.sum(Y_true == Y_pred)
    total_samples = len(Y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

# leeghalen voor uploaden naar github, voeg het pad naar het databestand toe en vervolgens na de komma sep=';' anders werkt het niet
nier = pd.read_csv()
nier_voorspellende_variabele = nier.drop(columns=[' vit_stat'])
nier_doelvariabele = nier[' vit_stat']

X = nier.iloc[:, :4]
Y = nier[' vit_stat']

num_classes = len(np.unique(Y))
Y_onehot = one_hot_encoding(Y, num_classes)

np.random.seed(42)

train_indices = np.random.choice(len(X), size=int(0.8 * len(X)), replace=False)

X_train = X.iloc[train_indices, :]
Y_train = Y_onehot[train_indices, :]

X_test = X.drop(train_indices, axis=0)
Y_test = Y_onehot[np.setdiff1d(np.arange(len(X)), train_indices), :]

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

def sigmoid(x):
    return 1 / (1 + np.e**-x)

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.uniform(-0.01, 0.01, size=(input_size, hidden_size))
    b1 = np.random.uniform(-0.01, 0.01, size=(1, hidden_size))
    W2 = np.random.uniform(-0.01, 0.01, size=(hidden_size, output_size))
    b2 = np.random.uniform(-0.01, 0.01, size=(1, output_size))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_propagation(X, params):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

def compute_loss(Y_true, Y_pred):
    epsilon = 1e-10
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(Y_true * np.log(Y_pred))
    return loss

def backward_propagation(X, Y_true, params, cache):
    m = X.shape[0]

    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']

    dZ2 = A2 - Y_true
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, params['W2'].T) * A1 * (1 - A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return gradients

def update_parameters(params, gradients, learning_rate):
    params['W1'] -= learning_rate * gradients['dW1']
    params['b1'] -= learning_rate * gradients['db1']
    params['W2'] -= learning_rate * gradients['dW2']
    params['b2'] -= learning_rate * gradients['db2']

    return params
