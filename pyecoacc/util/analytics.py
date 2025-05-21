import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


def compute_confusion_matrix(y_true, y_pred, normalize='true', round=2):
    lbls = list(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=lbls, normalize=normalize)
    return pd.DataFrame(cm, index=lbls, columns=lbls).round(round)


def compare_models_cv(X, y, model_dict, cv=5):
    reports = dict()
    overall_accuracy = dict()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    for model_name, clf in model_dict.items():
        print(f"Starting model {model_name}...")

        y_pred = cross_val_predict(clf, X, y_encoded, cv=cv, n_jobs=-1)
        y_pred = le.inverse_transform(y_pred)

        report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
        reports[model_name] = report

        overall_accuracy[model_name] = (y_pred == y).mean() * 100.

    reports = {model: pd.DataFrame(data) for model, data in reports.items()}
    models = reports.keys()

    accuracy = pd.Series(overall_accuracy)
    reports = {model: data.drop("accuracy", axis=1) for model, data in reports.items()}

    precision = pd.DataFrame([reports[model].loc["precision"] for model in models], index=models)
    recall = pd.DataFrame([reports[model].loc["recall"] for model in models], index=models)
    f1 = pd.DataFrame([reports[model].loc["f1-score"] for model in models], index=models)

    return accuracy, precision, recall, f1


