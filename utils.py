import numpy as np

def balanced_accuracy(y, y_pred):
    b_acc = (np.mean(y_pred[y == 0] == y[y == 0]) +\
         np.mean(y_pred[y == 1] == y[y == 1])) / 2
    return b_acc