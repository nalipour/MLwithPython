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

# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line


# Create a graph with input X plus a convolutional layer applying 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})


plt.imshow(output[0, :, :, 1], cmap='gray')  # plot 1st image's 2nd feature map
plt.show()
