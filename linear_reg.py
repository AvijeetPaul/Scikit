from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


# boston =datasets.load_boston()

#features /labels
X= data
y= target

# print(X)

#algorithm
l_reg = linear_model.LinearRegression()

plt.scatter(X.T[5],y)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#train
model= l_reg.fit(X_train,y_train)
predicts= model.predict(X_test)
print("Predictions: ", predicts)
print("R^2 value: ", l_reg.score(X,y))
print("coeff: ", l_reg.coef_)
print("intercept: ", l_reg.intercept_)
