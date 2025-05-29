
import numpy as np
import pandas as pd


def confusion_matrix_correction(budget, cm):
    corrected = np.dot(np.linalg.inv(cm).T, budget)
    return pd.Series(corrected, index=budget.index)


def compute_time_budget(raw_acc, clf, cm=None, apply_cm_correction=True):

    if apply_cm_correction and cm is None:
        raise ValueError("Confusion matrix must be provided if apply_cm_correction=True")

    y_hat = clf.predict(raw_acc)
    tb = pd.Series(y_hat).value_counts(normalize=True)

    if apply_cm_correction:
        tb = confusion_matrix_correction(tb[cm.index], cm)

    return tb

