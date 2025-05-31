from sklearn.model_selection import train_test_split

from .analytics import *


def train_compute_cm(model, X, y, cm_estimation_percent=.2, round=2):
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cm_estimation_percent)

    # train
    model.fit(X_train, y_train)

    # estimate
    y_hat = model.predict(X_test)
    cm = compute_confusion_matrix(y_test, y_hat, round=round)

    return cm


