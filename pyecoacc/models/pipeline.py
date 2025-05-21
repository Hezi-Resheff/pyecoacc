
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from skorch import NeuralNetClassifier


from ..features.transform import ACCStatsTransformer


def make_classifier_pipeline(features, model, feature_scaler=False, feature_selector=False, k_selection=25):

    steps = [
        ('features', features),
        ('model', model)
    ]

    if feature_scaler:
        scaling_step = ('scaler', StandardScaler())
        steps.insert(1, scaling_step)

    if feature_selector:
        selection_step = ('selection', SelectKBest(score_func=f_classif, k=k_selection))
        steps.insert(1, selection_step)

    return Pipeline(steps)


def get_default_random_forest_pipeline():
    model = RandomForestClassifier(n_estimators=250, max_depth=10)
    features = ACCStatsTransformer()
    return make_classifier_pipeline(features, model, feature_scaler=False, feature_selector=False)

