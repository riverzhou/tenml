#!/usr/bin/env python3

from sklearn import datasets
from sklearn import linear_model

#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays

boston = datasets.load_boston()

x_train=boston.data[:,5].reshape(-1,1)
y_train=boston.target.reshape(-1,1)
x_test=boston.data[:50,5].reshape(-1,1)

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)

#Equation coefficient and Intercept
print('Coefficient: n', linear.coef_)
print('Intercept: n', linear.intercept_)

#Predict Output
predicted= linear.predict(x_test)

print('predicted', predicted)

