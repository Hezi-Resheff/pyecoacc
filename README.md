# pyecoacc
A python package for supervised learning of behavioral modes from sensor data.

pyecoacc is built on top of sklearn and pytorch, and provides convenient pipelines of feature computations and other preprocessing necessary to easily run behavior classification on sensor data. This package contains easy to use out of the box optoins, together with extendability and customization. 


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

```python 
from pyecoacc.models.pipeline import get_default_random_forest_pipeline

classifier = get_default_random_forest_pipeline()
classifier.fit(ACC_train, y_train)
y_hat = classifier.predict(ACC_test)
```

