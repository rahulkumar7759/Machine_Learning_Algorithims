# Support Vector Machines (SVMs) are a powerful and widely used class of machine learning algorithms.
# They are particularly well-suited for classification tasks and have been successfully applied to a variety of domains,
# including image classification, text classification, and bioinformatics.

# In this algorithm, we will provide an introduction to SVMs, discuss how they work, and explore their Pros and cons

 What is a Support Vector Machine?

#               A Support Vector Machine is a type of supervised machine learning algorithm that can be used for classification or regression tasks. 
#               In a classification task, the SVM takes labelled data and learns a decision boundary that separates the different classes.
#              This decision boundary is called a hyperplane and is used to classify new, unlabelled data.

#              The key idea behind SVMs is to find the hyperplane that maximizes the margin between the different classes. 
#              The margin is the distance between the hyperplane and the closest data points from each class. 
#              The SVM tries to find the hyperplane that maximizes this margin, which is why it is also known as a maximum margin classifier.

#              In cases where the data is not linearly separable, SVMs can use a technique called kernel methods to map the data into a higher-dimensional space where
#               it is easier to separate the different classes.
#              This allows the SVM to still find a hyperplane that separates the data with a large margin.

 How does an SVM work?

#      An SVM works by taking a set of labelled training data and finding the hyperplane that maximizes the margin between the different classes.
#      The algorithm does this by solving an optimization problem, where the objective is to minimize the classification error while maximizing the margin.

#      The SVM works by constructing a hyperplane in a feature space with the largest possible margin between the two classes.
#       The points that are closest to the hyperplane are called support vectors, and they are used to define the hyperplane.

#      To classify new data, the SVM calculates the distance between the data point and the hyperplane. 
#      If the distance is positive, the point is classified as belonging to one class, and if the distance is negative, it is classified as belonging to the other class.

Pros and cons of SVMs: - 

#        SVMs have several strengths that make them a popular choice for machine learning tasks. 
#        One of the key strengths of SVMs is their ability to handle high-dimensional data.
#        They are also very effective in cases where the data is not linearly separable, thanks to their ability to use kernel methods to map the data into a higher-dimensional space.

#         However, SVMs also have some limitations. One of the biggest drawbacks of SVMs is that they can be computationally expensive to train, 
#         particularly when dealing with large datasets. Additionally, SVMs can be sensitive to the choice of kernel function,
#         and the performance of the algorithm can be heavily influenced by this choice.

Conclusion


#         Support Vector Machines are a powerful class of machine learning algorithms that are particularly well-suited for classification tasks.
#         They work by finding the hyperplane that maximizes the margin between the different classes, and they can use kernel methods to handle non-linearly separable data.

#          While SVMs have some limitations, they are widely used in a variety of domains and have proven to be effective in many applications.
#           By understanding the strengths and weaknesses of SVMs, machine learning practitioners can make informed decisions about when to use SVMs and how to tune them for
#           optimal performance.

# implementing the svm algorithm in text Classification 

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd 

# Load data
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(twenty_train.data)
y_train = twenty_train.target

# Define SVM classifier
clf = svm.SVC(kernel='linear', C=1.0)

# Train the SVM
clf.fit(X_train, y_train)

# Test the classifier on new data
twenty_test = fetch_





