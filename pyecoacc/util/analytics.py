import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def compute_confusion_matrix(y_true, y_pred, normalize='true', round=2):
    lbls = list(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=lbls, normalize=normalize)
    return pd.DataFrame(cm, index=lbls, columns=lbls).round(round)


def model_analytics_cv(X, y, model, cv=5, random_state=42):
    splits = dict()
    overall_accuracy = dict()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        y_hat = model.fit(X_train, y_train).predict(X_test)

        overall_accuracy[f"split_{i}"] = (y_hat == y_test).mean()

        report = classification_report(le.inverse_transform(y_test), le.inverse_transform(y_hat),
                                       target_names=le.classes_,
                                       output_dict=True)
        report = pd.DataFrame(report)
        report.drop("accuracy", axis=1, inplace=True)
        splits[f"split_{i}"] = report

    mean_report = pd.concat(splits.values()).groupby(level=0).mean()
    std_report = pd.concat(splits.values()).groupby(level=0).std()

    return overall_accuracy, mean_report, std_report, splits


def compare_models_cv(X, y, model_dict, cv=5, round_digits=3):
    all_data = dict()
    accuracy = dict()

    for model_name, clf in model_dict.items():
        print(f"Starting model {model_name}...")

        model_accuracy, mean_report, std_report, splits = model_analytics_cv(X, y, clf, cv=cv)

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



