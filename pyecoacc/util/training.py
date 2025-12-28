from sklearn.model_selection import train_test_split

from .analytics import compute_confusion_matrix


def train_compute_cm(model, X, y, cm_estimation_percent=.2, round=2):
    """_summary_

    Args:
        model (Pipeline or Estimator): the model to train and evaluate
        X (np.array): featuers to train and evaluate on
        y (np.array): labels to train and evaluate on
        cm_estimation_percent (float, optional): the fraction of data used for confusion matrix estimation. Defaults to .2.
        round (int, optional): the number of decimal places to round the confusion matrix. Defaults to 2.

    Returns:
        confusion_matrix (pd.DataFrame): the estimated confusion matrix
    """
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cm_estimation_percent)

    # train
    model.fit(X_train, y_train)

    # estimate
    y_hat = model.predict(X_test)
    cm = compute_confusion_matrix(y_test, y_hat, round=round)

    return cm


