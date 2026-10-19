import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as K
from keras import layers
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

def one_hot_encoding(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def calculate_accuracy(Y_true, Y_pred):
    correct_predictions = np.sum(Y_true == Y_pred)
    total_samples = len(Y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

keras = pd.read_csv('/Users/femkeboersma/Downloads/iknl_nier_syntheticdata.csv', sep=";")
keras.columns = keras.columns.str.strip()

keras_voorspellers = ['leeft', 'gesl', 'tumsoort', 'diag_basis', 'gedrag','tum_afm', 'later', 'topo_sublok', 'pos_lymf']
keras_doelvariabele = 'vit_stat'

keras = keras[keras_voorspellers + [keras_doelvariabele]].dropna()

categorisch = ['topo_sublok']
numeriek = [k for k in keras_voorspellers if k not in categorisch]

keras['topo_sublok'] = keras['topo_sublok'].astype(str).str.strip()
codes, categorieen = pd.factorize(keras['topo_sublok'])
topo_onehot = one_hot_encoding(codes, len(categorieen))

X = np.hstack([keras[numeriek].to_numpy(dtype=float), topo_onehot])
Y = keras[keras_doelvariabele].to_numpy(dtype=int)

print(keras.columns.tolist())
print(X.shape, X.dtype)

num_classes = len(np.unique(Y))
Y_onehot = one_hot_encoding(Y, num_classes)

np.random.seed(42)

train_indices = np.random.choice(len(X), size=int(0.8 * len(X)), replace=False)
test_indices = np.setdiff1d(np.arange(len(X)), train_indices)

X_train = X[train_indices, :]
Y_train = Y_onehot[train_indices, :]

X_test = X[test_indices, :]
Y_test = Y_onehot[test_indices, :]

# --- Normaliseren (z-score), alleen berekend op trainingsdata ---
X_test_ruw = X_test.copy()            # ruwe waarden bewaren voor de controle straks
gemiddelde = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1                     # voorkomt delen door 0 als een kolom constant is
X_train = (X_train - gemiddelde) / std
X_test = (X_test - gemiddelde) / std

def sigmoid(x):
    return 1 / (1 + np.e**-x)

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.uniform(-0.5, 0.5, size=(input_size, hidden_size))
    b1 = np.random.uniform(-0.5, 0.5, size=(1, hidden_size))
    W2 = np.random.uniform(-0.5, 0.5, size=(hidden_size, output_size))
    b2 = np.random.uniform(-0.5, 0.5, size=(1, output_size))

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

print(X_train.dtype, X_train.shape)

def train_mlp(X_train, Y_train, input_size, hidden_size, output_size, learning_rate, epochs):
    params = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(1, epochs + 1):
        cache = forward_propagation(X_train, params)
        Y_pred = cache['A2']

        loss = compute_loss(Y_train, Y_pred)

        gradients = backward_propagation(X_train, Y_train, params, cache)

        params = update_parameters(params, gradients, learning_rate)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    return params


np.random.seed(42)

input_size = X_train.shape[1]
print(input_size)
hidden_size = 16
output_size = Y_train.shape[1]
learning_rate = 1.0
epochs = 5000

trained_params = train_mlp(X_train, Y_train, input_size, hidden_size, output_size, learning_rate, epochs)

def predict_mlp(X, params):
    cache = forward_propagation(X, params)
    Y_pred = cache['A2']

    predictions = np.argmax(Y_pred, axis=1)

    return predictions

Y_pred_test = predict_mlp(X_test, trained_params)

Y_test_labels = np.argmax(Y_test, axis=1)
accuracy = calculate_accuracy(Y_test_labels, Y_pred_test)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

W2 = trained_params['W2']
b2 = trained_params['b2'].ravel()

keras_model = K.Sequential([
    K.Input(shape=(input_size,)),
    layers.Dense(hidden_size, activation='sigmoid'),
    layers.Dense(1, activation='sigmoid'),
])

keras_model.set_weights([
    trained_params['W1'],
    trained_params['b1'].ravel(),
    (W2[:, 1] - W2[:, 0]).reshape(-1, 1),
    np.array([b2[1] - b2[0]]),
])

keras_pred = (keras_model.predict(X_test, verbose=0)[:, 0] >= 0.5).astype(int)
print("Zelfde voorspellingen als ons eigen netwerk:", np.array_equal(keras_pred, Y_pred_test))

keras_model.save("model.keras")
print("Model opgeslagen als model.keras")

def predict_proba(X, params):
    return forward_propagation(X, params)['A2'][:, 1]

def train_mlp(X_tr, Y_tr, X_va, Y_va, hidden_size, learning_rate,
              max_epochs=5000, patience=100, verbose=False):
    params = initialize_parameters(X_tr.shape[1], hidden_size, Y_tr.shape[1])
    best, best_params, wait = np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        cache = forward_propagation(X_tr, params)
        grads = backward_propagation(X_tr, Y_tr, params, cache)
        params = update_parameters(params, grads, learning_rate)

        val_loss = compute_loss(Y_va, forward_propagation(X_va, params)['A2'])
        if val_loss < best - 1e-5:
            best, wait = val_loss, 0
            best_params = {k: v.copy() for k, v in params.items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}, val loss {val_loss:.4f}")
    return best_params, best, epoch

# Validatieset uit de trainingsdata (de testset blijft ongemoeid)
rng = np.random.default_rng(42)
idx = rng.permutation(len(X_train))
n_val = int(0.2 * len(idx))
val_idx, fit_idx = idx[:n_val], idx[n_val:]
X_fit, Y_fit = X_train[fit_idx], Y_train[fit_idx]
X_val, Y_val = X_train[val_idx], Y_train[val_idx]
y_val = Y_val.argmax(axis=1)

print("Baseline accuracy (altijd meest voorkomende klasse):",
      max(y_val.mean(), 1 - y_val.mean()))

resultaten = []
for hidden_size in [3, 8, 16, 32, 64]:
    for learning_rate in [0.05, 0.1, 0.5, 1.0]:
        params, val_loss, n_ep = train_mlp(X_fit, Y_fit, X_val, Y_val, hidden_size, learning_rate)
        p = predict_proba(X_val, params)
        auc = roc_auc_score(y_val, p)
        resultaten.append((auc, val_loss, hidden_size, learning_rate, n_ep))
        print(f"hidden={hidden_size:3d} lr={learning_rate:<5} epochs={n_ep:4d} "
              f"val loss={val_loss:.4f} AUC={auc:.3f} Brier={brier_score_loss(y_val, p):.3f}")

resultaten.sort(reverse=True)
beste_auc, _, hidden_size, learning_rate, beste_epochs = resultaten[0]
print("Beste:", hidden_size, learning_rate, "AUC", round(beste_auc, 3))

trained_params, _, _ = train_mlp(X_fit, Y_fit, X_val, Y_val, hidden_size, learning_rate)
Y_pred_test = predict_mlp(X_test, trained_params)
p_test = predict_proba(X_test, trained_params)
y_test_labels = Y_test.argmax(axis=1)
print("Test AUC:  ", round(roc_auc_score(y_test_labels, p_test), 3))
print("Test Brier:", round(brier_score_loss(y_test_labels, p_test), 3))
print("Test acc:  ", round(calculate_accuracy(y_test_labels, Y_pred_test), 3))