import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder


def compute_confusion_matrix(y_true, y_pred, normalize='true', round=2):
    lbls = list(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=lbls, normalize=normalize)
    return pd.DataFrame(cm, index=lbls, columns=lbls).round(round)


def model_analytics_cv(X, y, model, cv=5, cv_method="stratified", individuals=None,
                       random_state=42):
    splits_output = dict()
    overall_accuracy = dict()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    split_indices = None

    if cv_method == "stratified":
        cross_val_spliter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        split_indices = cross_val_spliter.split(X, y)

    elif cv_method == "LOIO":
        cross_val_splitter = LeaveOneGroupOut()
        split_indices = cross_val_splitter.split(X, y, individuals)

    else:
        raise ValueError(f"Unsupported cross-validation method: {cv_method}")

    for i, (train_index, test_index) in enumerate(split_indices):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        y_hat = model.fit(X_train, y_train).predict(X_test)

        group_name = None
        if cv_method == "stratified":
            group_name = i + 1
        elif cv_method == "LOIO":
            group_name = np.unique(individuals[test_index])[0]

        overall_accuracy[f"split-{group_name}"] = (y_hat == y_test).mean()

        report = classification_report(le.inverse_transform(y_test), le.inverse_transform(y_hat),
                                       labels=le.classes_,
                                       output_dict=True,
                                       zero_division=0)
        report = pd.DataFrame(report)
        report.drop("accuracy", axis=1, inplace=True)
        splits_output[f"split-{group_name}"] = report

    mean_report = pd.concat(splits_output.values()).groupby(level=0).mean()
    std_report = pd.concat(splits_output.values()).groupby(level=0).std()

    return overall_accuracy, mean_report, std_report, splits_output


def compare_models_cv(X, y, model_dict, cv=5, cv_method="stratified", individuals=None,
                      random_state=42, round_digits=3):
    all_data = dict()
    accuracy = dict()

    for model_name, clf in model_dict.items():
        print(f"Starting model {model_name}...")

        model_accuracy, mean_report, std_report, splits = model_analytics_cv(X, y, clf,
                                                                             cv=cv,
                                                                             cv_method=cv_method,
                                                                             individuals=individuals,
                                                                             random_state=random_state)

        all_data[model_name] = {"mean_report": mean_report, "std_report": std_report, "splits": splits}
        accuracy[model_name] = model_accuracy

    # Overall
    accuracy = pd.DataFrame(accuracy)
    accuracy.loc["mean"] = accuracy.mean().rename('mean')
    accuracy.loc["std"] = accuracy.std().rename('std')

    # Recall, Precision, F1
    mean_std_reports = {name: info["mean_report"].round(round_digits).astype(str) + " (" + info["std_report"].round(round_digits).astype(str) + ")"
                        for name, info in all_data.items()}
    recall = pd.DataFrame({name: frame.loc["recall"] for name, frame in mean_std_reports.items()})
    precision = pd.DataFrame({name: frame.loc["precision"] for name, frame in mean_std_reports.items()})
    f1 = pd.DataFrame({name: frame.loc["f1-score"] for name, frame in mean_std_reports.items()})

    return accuracy, recall, precision, f1, all_data



