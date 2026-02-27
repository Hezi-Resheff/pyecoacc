# pyecoacc
A python package for supervised learning of behavioral modes from sensor data.

pyecoacc is built on top of sklearn and pytorch, and provides convenient pipelines of feature computations and other preprocessing necessary to easily run behavior classification on sensor data. This package contains easy to use out of the box optoins, together with extendability and customization. 

## Documentation
You can find the full documentation [here](https://hezi-resheff.github.io/pyecoacc/).
Complete workflow examples can be found in the examples folder. 

## 📦 Installation

Install from PyPI: 


```bash
pip install pyecoacc   
```

Or from the source:


```bash
git clone https://github.com/Hezi-Resheff/pyecoacc.git
cd pyecoacc
pip install .
```

#### 🔗 Dependencies
All required dependencies are listed in requirements.txt. To install them manually:

```bash
pip install -r requirements.txt
```


## 🧑‍💻 Usage

pyecoacc allows a wide range of classifiers ranging from simple out of the box models, to custom pipelines and deep learning. 
 
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

Custom CNN classifier (don't actually use anything so small unless it's just as a quick baseline...):

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

## 📖 Citation
If you use pyecoacc in your research, please cite it using the following BibTeX entry:

@article{resheff2026pyecoacc,
  title={pyecoacc: A python package for supervised learning of behavioural modes from accelerometer data},
  author={Resheff, Yehezkel S and Harel, Roi and Zlotnick, Omer B and Rotics, Shay},
  journal={Methods in Ecology and Evolution},
  year={2026},
  publisher={Wiley Online Library}
}
