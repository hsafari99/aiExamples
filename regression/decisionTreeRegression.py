import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[ : , 1:-1].values
y = dataset.iloc[ : , -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

from sklearn.metrics import r2_score
r_val = r2_score(y_true=y, y_pred=regressor.predict(X))
print(r_val)

# print(regressor.predict([[6.5]]))