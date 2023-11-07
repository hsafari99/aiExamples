import numpy as np
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[ : , 1:-1 ].values
y = dataset.iloc[ : , -1 ].values

from sklearn.ensemble import RandomForestRegressor
resgressor = RandomForestRegressor(n_estimators=10, random_state=0)
resgressor.fit(X, y)

from sklearn.metrics import r2_score
r_val = r2_score(y_true=y, y_pred=resgressor.predict(X))
print(r_val)