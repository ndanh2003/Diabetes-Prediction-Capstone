from ucimlrepo import fetch_ucirepo 
import pandas as pd

cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 

X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

full_data = pd.concat([X, y], axis=1)
full_data.to_csv('../data/cdc_dataset.csv', index=False)