
import numpy as np
import pandas as pd


def confusion_matrix_correction(budget, cm):
    """Apply the confusion matrix correction for time budgets. 

    Args:
        budget (pd.Series): the raw time budget
        cm (pd.DataFrame): the confusion matrix used for correction

    Returns:
        corrected_budget (pd.Series): the corrected time budget
    """
    corrected = np.dot(np.linalg.inv(cm).T, budget)
    return pd.Series(corrected, index=budget.index)


def compute_time_budget(raw_acc, clf, cm=None, apply_cm_correction=True):
    """Use a trained classifier to compute a time budget. 

    Args:
        raw_acc (np.array): the raw accelerometer data to compute the time budget from
        clf (Pipeline): the trained classifier
        cm (pd.DataFrame, optional): the confusion matrix used for correction. Defaults to None.
        apply_cm_correction (bool, optional): If False, no correction is applied. Defaults to True.


    Returns:
        budget (pd.Series): the time budget
    """

    if apply_cm_correction and cm is None:
        raise ValueError("Confusion matrix must be provided if apply_cm_correction=True")

    y_hat = clf.predict(raw_acc)
    tb = pd.Series(y_hat).value_counts(normalize=True)

    if apply_cm_correction:
        tb = confusion_matrix_correction(tb[cm.index], cm)

    return tb

