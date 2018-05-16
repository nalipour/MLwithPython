import tensorflow as tf
import numpy as np


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_epochs = 20
batch_size = 200

saver = tf.train.import_meta_graph('./my_model_final.ckpt.meta')


# Get operations
for op in tf.get_default_graph().get_operations():
    print(op.name)

# Possibility to visualise the graph: check GitHub of Aurelien Geron

X = tf.get_default_graph().get_tensor_by_name('X:0')
y = tf.get_default_graph().get_tensor_by_name('y:0')

accuracy = tf.get_default_graph().get_tensor_by_name('eval/accuracy:0')
training_op = tf.get_default_graph().get_operation_by_name('GradientDescent')

# Regroup important operations to make the access easier
for op in (X, y, accuracy, training_op):
    tf.add_to_collection('my_important_ops', op)

X, y, accuracy, training_op = tf.get_collection('my_important_ops')

with tf.Session() as sess:
    saver.restore(sess, './my_model_final.ckpt')

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, 'Validation accuracy:', accuracy_val)

    save_path = saver.save(sess, './reuseTensorflow.ckpt')
