# Welcome to PyEcoACC


A python package for supervised learning of behavioral modes from accelerometer data.

pyecoacc is built on top of sklearn and pytorch, and provides convenient pipelines of feature computations and other preprocessing necessary to easily run behavior classification on sensor data. This package contains easy to use out of the box optoins, together with extendability and customization. 


## 📦 Installation

Install from PyPI: 


```bash
pip install pyecoacc   
```


## 🧑‍💻 Usage

PyEcoACC allows a wide range of classifiers ranging from simple out of the box models, to custom pipelines and deep learning. 
 
Simple out of the box classifier:

```python 
from pyecoacc.models.pipeline import get_default_random_forest_pipeline

classifier = get_default_random_forest_pipeline()
classifier.fit(ACC_train, y_train)
y_hat = classifier.predict(ACC_test)
```

Simple XGBoost based model:

```python 
from xgboost import XGBClassifier
from pyecoacc.models.pipeline import make_classifier_pipeline

xg_model = make_classifier_pipeline(features=ACCStatsTransformer(), 
                                    model=XGBClassifier(n_estimators=250)) 
```

Custom CNN classifier:

```python 
from pyecoacc.models.deep.cnn import make_cnn_model


tiny_cnn_model = make_cnn_model(input_dim=INPUT_ACC_LENGTH, 
                                num_behav=NUMBER_OF_BEHAVIORS, 
                                conv_filters=[32, 64],
                                fc_layers=[20],
                                verbose=0)

classifier.fit(ACC_train, y_train)
y_hat = classifier.predict(ACC_test)
```