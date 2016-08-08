#!/usr/bin/env python3

from sklearn import datasets
from sklearn import linear_model

import numpy as np

import matplotlib.pyplot as plt

#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays

boston = datasets.load_boston()

x_train=boston.data[:,5,np.newaxis]
y_train=boston.target
x_test=boston.data[:50,5,np.newaxis]
y_test=boston.target[:50]

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
print('Regress score :', linear.score(x_train, y_train))

#Equation coefficient and Intercept
print('Coefficient: n', linear.coef_)
print('Intercept: n', linear.intercept_)

#Predict Output
predicted= linear.predict(x_test)

for i in range(len(x_test)) :
        print(x_test[i], y_test[i], predicted[i])
#print('predicted', predicted)

plt.scatter(y_test,predicted)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()])
plt.show()

