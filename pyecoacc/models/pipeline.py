
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from skorch import NeuralNetClassifier


from ..features.transform import ACCStatsTransformer


def make_classifier_pipeline(features, model, feature_scaler=False, feature_selector=False, k_selection=25):
    """Make a custom classification pipeline with optional feature scaling and selection.

    Args:
        features (sklearn.Transformer): transforms the raw data into features, for example the pyecoacc ACCStatsTransformer object.
        model (sklearn.Estimator): the classification model to use. For example, an sklearn RandomForestClassifier object. 
        feature_scaler (bool, optional): If True, applies StandardScaler to the features. Defaults to False.
        feature_selector (bool, optional): if True, applies SelectKBest to the features. Defaults to False.
        k_selection (int, optional): The number of top features to select. Defaults to 25.

    Returns:
        Model (Pipeline): A pipeline object with the specified model.
    """

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
    """
    Builds a default Random Forest classification pipeline (250 trees) with ACC stats features.
    
    Returns:
        Model (Pipeline): A pipeline object with the default model.
    """
    model = RandomForestClassifier(n_estimators=250, max_depth=10)
    features = ACCStatsTransformer()
    return make_classifier_pipeline(features, model, feature_scaler=False, feature_selector=False)

