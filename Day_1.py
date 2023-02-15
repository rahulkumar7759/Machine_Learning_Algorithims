# K-NN algorithims:-  

# K-Nearest Neighbors (KNN) is a simple and effective machine learning algorithm used for both regression and classification tasks. 
# It's a non-parametric algorithm, which means that it does not make any assumptions about the underlying distribution of the data. 
# In this blog, we'll explore how KNN works and how to implement it in Python.

# HOW KNN WORKS:- 

# KNN is a lazy learning algorithm, which means that it does not learn a model from the training data but rather stores the training data itself.
# To make predictions on new data points,
# KNN looks for the K nearest neighbors to the new point in the training data and predicts the output based on the most common output among those K neighbors.
# The value of K is a hyperparameter that needs to be tuned for optimal performance.KNN is distance-based, which means that it calculates the distance between the new point and all the training data points using a distance metric such as Euclidean or Manhattan distance. 
# The distance between two points can be thought of as the similarity between them, with smaller distances indicating greater similarity.

# Implementing KNN in Python

# To illustrate how KNN works, let's implement it on a sample dataset.
# We'll use the scikit-learn library to load the iris dataset, which consists of 150 samples of iris flowers,
# with four features each: sepal length, sepal width, petal length, and petal width. The goal is to predict the species of the iris flower based on these features.
# First, we'll split the dataset into training and testing sets:

# code :- 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Next, we'll import the KNeighborsClassifier class from scikit-learn and create an instance of it:
# code :- 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

# We set n_neighbors to 3, which means that KNN will look for the 3 nearest neighbors to a new point to make a prediction.
# We then fit the model to the training data:
  
  knn.fit(X_train, y_train)

 # Now, we can make predictions on the test data and evaluate the performance of the model using metrics such as accuracy and confusion matrix:
 
# code : -

from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Conclusion:- 

  # KNN is a simple yet powerful machine learning algorithm that can be used for both regression and classification tasks.
  # It works by finding the K nearest neighbors to a new point in the training data and making a prediction based on the most common output among those K neighbors.
  # KNN is easy to implement in Python using the scikit-learn library, and its performance can be evaluated using metrics such as accuracy and confusion matrix. 
  # With its simplicity and effectiveness, KNN is a great algorithm to have in your machine learning toolbox.
  
  
