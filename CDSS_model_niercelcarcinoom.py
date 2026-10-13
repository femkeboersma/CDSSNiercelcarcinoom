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


