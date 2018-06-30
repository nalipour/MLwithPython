from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

import numpy as np
import tensorflow as tf

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length and petal width
y = (iris.target == 0).astype(np.int)  # Iris setosa ?

# ### Perceptron ### #
# Perceptron is equivalent to Stochastic Gradient Descent (SGDClassifier)
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])


# ### FNN (Feedforward Neural Network) ### #
# MINST Data set
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# ---- High-level API
feature_cols = [tf.feature_column.numeric_column('X', shape=[28*28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                     feature_columns=feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(x={'X': X_train}, y=y_train,
                                              num_epochs=40, batch_size=50,
                                              shuffle=True)
dnn_clf.train(input_fn=input_fn)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'X': X_test}, y=y_test, shuffle=False)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
y_pred = list(y_pred_iter)
y_pred[0]
