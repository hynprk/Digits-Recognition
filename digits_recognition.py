"""
Created on Fri Aug 12 23:20:28 2022

@author: hyoeungracepark
"""

# Import library
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load data set
digits = datasets.load_digits()

# Print the keys
# DESCR,
# shape of the images,
# and data keys of the digits dataset
print(digits.keys())
print(digits.DESCR)
print(digits.images.shape)
print(digits.data.shape)

# Display digit 408
#plt.imshow(digits.images[408], cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()

########

# Creating arrays for features and response variable
X = digits.data
y = digits.target

# Split into train and test data sets
## stratify = y: distributed in train and test sets  
## as they were in the original data set
X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 42, 
                                                    stratify = y)

# Create k-NN classifier with 6 neighbours
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit classifier to the training data
knn.fit(X_train, y_train)

# Print accuracy of classifier's prediction using knn.score()
pred_acc = knn.score(X_test, y_test)
print("Testing Accuracy when k-NN = 6: {}".format(pred_acc))

### Testing a diverse number of k-neighbours

# Arrays to store train and test set accuracies
neighbours = np.arange(1, 15)
train_accuracy = np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

# Loop over different values of k
for i, k in enumerate(neighbours):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot with matplotlib.pyplot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbours, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbours, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


