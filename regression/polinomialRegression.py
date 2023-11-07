import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[ : , 1: -1 ].values
y = dataset.iloc[ : , -1 ].values

from sklearn.linear_model import LinearRegression
line_reg = LinearRegression()
line_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
line_reg_2 = LinearRegression()
line_reg_2.fit(X_poly, y)

test = line_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(test)
# plt.scatter(X, y, color='red')
# plt.plot(X, line_reg_2.predict(X_poly), color='blue')
# plt.title("Linear regression results")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()