
import numpy as np


def confusion_matrix_correction(budget, cm):
    return np.dot(np.linalg.inv(cm).T, budget)

