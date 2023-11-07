import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data from file 
dataset = pd.read_csv("50_Startups.csv")

# Get all column values except the last one and put it in X value
# Get all row values for last column and put it in y
X = dataset.iloc[ : , : -1].values
y = dataset.iloc[ : , -1 ].values

# Handle state categorization
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Define test and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model using BACKWARD ELIMINATION technique
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X=X_train, y=y_train)

y_pred = regressor.predict(X=X_test)
np.set_printoptions(precision=2)
real_vs_pred = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1)

plt.scatter(y_pred, y_test, color='blue')
plt.title("Predict Vs. Real Data")
plt.xlabel("Predict")
plt.ylabel("Actual data")
plt.show()
print(real_vs_pred)
