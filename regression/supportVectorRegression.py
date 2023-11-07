import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Put 2nd column (level) in X
# Pu salaries in y
X = dataset.iloc[ : , 1:-1 ].values
y = dataset.iloc[ : , -1].values

# Reshape y to be vertical array
y = y.reshape(len(y), 1)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1)))

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color='blue')
plt.title("SVR Results")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


print(y)