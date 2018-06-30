from sklearn.datasets import load_sample_image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

%matplotlib inline

# Load sample images
china = load_sample_image('china.jpg')
flower = load_sample_image('flower.jpg')
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# Create a graph with input X plus a convolutional layer applying 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='VALID')

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

plt.imshow(china)
plt.imshow(output[0].astype(np.uint8))

np.shape(china)
np.shape(output[0])
