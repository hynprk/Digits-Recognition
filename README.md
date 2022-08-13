# Digits Recognition Tutorial with MNIST Data (Python)

This is a basic supervised learning example using the digits dataset from sklearn in Python. Code without texts in between can be found in the file named [digits_recognition.py](https://github.com/hynprk/Digits-Recognition). I used [Spyder](https://www.spyder-ide.org/) to program in Python, but of course, Python can be used in various software tools. Please let me know if there are any errors in the information given below.

## Introduction
Machine learning is one of the important fields data scientist must know. Machine learning is a subfield of artificial intelligence (Yes, they are not the same!) since AI does not always imply a learning-based system. AI refers to machines executing tasks efficiently, while machine learning is programming algorithms that can automatically learn from data or from experience. Oftentimes, we cannot always execute the appropriate outcome by hand, especially if there is a wide range of abstract data we have to handle. This includes being able to recognize human speech, objects, handwriting, or people from pictures. 

There are three types of machine learning, which include:

* **Supervised Learning**: A machine is given data, including the examples to predict. In other words, the machine is given inputs (x) and labels (y). Here, the goal would be developing an algorithm to predict labels with high accuracy. Therefore, supervised learning is recommended when the data provides many examples of accurate predictions. 
* **Unsupervised Learning**: A machine is given data *without* the examples to predict.
* **Reinforcement Learning**: A machine obtains data by interacting with an environment and attempts to learn the optimal behaviour in that environment  to minimize cost.

## MNIST Data set

The MNIST (Modified National Institute of Standards and Technology) data set contains handwritten digits, which can be loaded by `load_digits` in `sklearn.datasets`. This is an extremely useful dataset if you are interested in getting beginners-friendly hands-on experience with machine learning. 

We require the following libraries for this mini supervised learning project:

```ruby
# Import libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
```
* [pandas](https://pandas.pydata.org/) is typically used for data wrangling and data analysis in Python. Its conventional alias is `pd`.
* [sklearn](https://scikit-learn.org/stable/) is a machine learning library in Python, which features algorithms such as regression, classification, and clustering. 
* [matplotlib](https://matplotlib.org/) is used for data visualization in Python. Here, we used `KNeighborsClassifier` which uses the k-Nearest Neighbours Classification method. Here, "k" is the number of nearest neighbours. Read [this](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn) to learn more about the k-Nearest Neighbours algorithm. 
* [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) ("Numerical Python") is used for working with arrays in Python. Python arrays are like lists but the stored elements are homogeneous (of the same type) and they can compute mathematical operations more efficiently and faster, whereas lists are less compact and cannot run mathematical operations in some cases. Read [this](https://stackoverflow.com/questions/176011/python-list-vs-array-when-to-use) or [this](https://stackoverflow.com/questions/993984/what-are-the-advantages-of-numpy-over-regular-python-lists) to learn more about the difference between lists and arrays in Python. The conventional alias for `numpy` is `np`. 

After importing the libraries, begin with loading the MNIST data set and store it as `digits` by using the `=` operator.

```ruby
# Load data set
digits = datasets.load_digits()
```
Before getting into the actual kNN method, we might want to take a look at some of the meta data (data of the dataset).

```ruby
# Print the keys
# DESCR,
# shape of the images,
# and data keys of the digits dataset
print(digits.keys())
print(digits.DESCR)
print(digits.images.shape)
print(digits.data.shape)
```

* `digits.keys()` executes the variables stored in the `digits` data. The keys are:

```
dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
```

* `digits.DESCR` loads the description of the `digits` data set. Note that `DESCR` is one of the keys as shown above. See [here](https://github.com/hynprk/Digits-Recognition/blob/main/DESCR.txt) in case you are interested in reading the description. 
* `digits.images.shape` and `digits.data.shape` executes shapes of `digits.image` and `digits.data`, respectively. `shape` in `pandas` returns a tuple with a number of elements for each dimension of the array. For instance, if we have
```ruby
ex_array = np.array([[[[[1, 2, 3, 4, 5]]]]])
```
then, `ex_array.shape` would return a tuple `(1, 1, 1, 1, 5)` since the first four dimensions have one element and the last dimension (fifth one) has five elements. To give another example, say we have the following array:
```ruby
ex_array2 = np.array([[[[1]]]], [[[[2]]]], [[[[3]]]])
```
Then, `print(ex_array.shape)` would print `(3, 1, 1, 1)` since the first dimension has four elements whereas the second, third, and fourth dimensions have only one element.

Hence, `print(digits.images.shape)` and `print(digits.data.shape)` print the tuples given below:
```
(1797, 8, 8)
(1797, 64)
```
This means that `digits.images` and `digits.data` are 3-D and 2-D arrays, respectively. There are 1797 data points in the MNIST data set.


We can also plot some of the digit data as images using `plt.imshow()` and `plot.show()`. 
```ruby
# Display digit 408
plt.imshow(digits.images[408], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
# Display digit 1008
plt.imshow(digits.images[1008], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
```

<p align="center">
  <img width = "30%" src="https://github.com/hynprk/Digits-Recognition/blob/main/vis/digit_no408.png">
  <img width = "30%" scr="https://github.com/hynprk/Digits-Recognition/blob/main/vis/digit_no1008.png">
</p>

```ruby
# Creating arrays for features and response variable
X = digits.data
y = digits.target

# Split into train and test data sets
## stratify = y: distributed in train and test sets  
## as they were in the original data set
X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 42, 
                                                    stratify = y)
```

```ruby
# Create k-NN classifier with 6 neighbours
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit classifier to the training data
knn.fit(X_train, y_train)

# Print accuracy of classifier's prediction using knn.score()
pred_acc = knn.score(X_test, y_test)
print("Testing Accuracy when k-NN = 6: {}".format(pred_acc))
```
```ruby
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
```

```ruby
# Generate plot with matplotlib.pyplot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbours, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbours, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```
