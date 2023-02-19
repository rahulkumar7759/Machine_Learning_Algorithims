
Introduction to Linear Regression

#      Linear regression is a supervised learning algorithm that is used for predicting the value of a numerical target variable based on one or more input variables.
#      The algorithm assumes a linear relationship between the input variables and the target variable.
#      The goal of linear regression is to find a linear function that best fits the training data. This function can then be used to make predictions on new data.

Simple Linear Regression:- 

#  Simple linear regression is used when there is only one input variable. The equation for simple linear regression is

y = mx + b
where y is the target variable, x is the input variable, m is the slope of the line, and b is the y-intercept.

The goal of simple linear regression is to find the values of m and b that best fit the training data.

Multiple Linear Regression

Multiple linear regression is used when there are multiple input variables. The equation for multiple linear regression is

y = b0 + b1*x1 + b2*x2 + ... + bn*xn
where y is the target variable, x1, x2, ..., xn are the input variables, and b0, b1, b2, ..., bn are the coefficients.

The goal of multiple linear regression is to find the values of b0, b1, b2, ..., bn that best fit the training data.

Implementation of Linear Regression
Now that we have a basic understanding of linear regression, let's see how to implement it using Python's Scikit-Learn library.

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
boston = load_boston()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

In this example, we load the Boston Housing dataset using Scikit-Learn's load_boston function. We then split the data into training and testing sets using 

Scikit-Learn's train_test_split function.

We create a linear regression model using Scikit-Learn's LinearRegression class and fit the model to the training data using the fit method.

We then make predictions on the testing data using the predict method and calculate the mean squared error and R^2 score using Scikit-Learn's mean_squared_error
and r2_score functions.

Finally, 
we print the results.

The mean squared error measures the average squared difference between the predicted and actual values. 
The R^2 score measures the proportion of variance in the target variable that can be explained by the input variables.


