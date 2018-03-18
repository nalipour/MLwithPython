from sklearn.datasets import fetch_mldata
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']
X.shape
y.shape

# Display an image (28x28)
some_digit = X[1]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.axis('off')
y[1]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Shuffle the data set
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# ####### Training a Binary Classifier ####### #
# Identify only one digit: 5-detector
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Stochastic Gradient Descent
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])

# ####### Performance Measures ####### #
# Cross validation implementation
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    '''
    print(X_train_folds.shape)
    print(y_train_folds.shape)
    print(X_test_fold.shape)
    print(y_test_fold.shape)
    '''

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')


# Classifier that classifies every image in the not-5 class
class Never5Class(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Class()
# About 10% of images are 5s ----> accuracy = 90 %: not a good measure
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy')

# ####### Confusion Matrix ####### #
# nb. of times instances of class A are classified as B
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)

# ####### Precision and Score ####### #
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
# F1-score favours classifiers with similar precision and recall
f1_score(y_train_5, y_train_pred)

# ####### Precision/Recall Tradeoff ####### #
y_scores = sgd_clf.decision_function([some_digit])
threshold = 23440
y_some_digit_pred = (y_scores > threshold)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

plt.plot(recalls, precisions)

# 90% precision
y_train_pred_90 = (y_scores > 70000)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)
