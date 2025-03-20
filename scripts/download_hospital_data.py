from ucimlrepo import fetch_ucirepo 
import pandas as pd
import warnings

try:
    from pandas.errors import DtypeWarning
    warnings.filterwarnings('ignore', category=DtypeWarning)
except ImportError:
    try:
        from pandas.core.common import DtypeWarning
        warnings.filterwarnings('ignore', category=DtypeWarning)
    except (ImportError, AttributeError):
        warnings.filterwarnings('ignore', message='.*Columns.*have mixed types.*')

diabetes_130_us_hospitals = fetch_ucirepo(id=296) 

X = diabetes_130_us_hospitals.data.features
y = diabetes_130_us_hospitals.data.targets

full_data = pd.concat([X, y], axis=1)
full_data.to_csv('../raw_data/hospital_dataset.csv', index=False)
