# Objective: demonstration of linear regression in Python for machine learning
# Source: Siraj Raval, How to Make a Prediction - Intro to Deep Learning #1
# Notes: script prepared for Python 3.6.0


import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
dataframe = pd.read_csv('.\data\data.csv', names=['Scores', 'Hours'])
print("Show first few lines of imported dataframe:")
print(dataframe.head())
x_values = dataframe[['Scores']]
y_values = dataframe[['Hours']]

# train model on data
scores_reg = linear_model.LinearRegression()
scores_reg.fit(x_values, y_values)

# visualise results
plt.scatter(x_values, y_values)
plt.plot(x_values, scores_reg.predict(x_values))
plt.show()