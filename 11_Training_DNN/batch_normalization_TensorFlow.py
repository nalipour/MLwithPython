from functools import partial

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

# A placeholder op that passes through input when its output is not fed.
training = tf.placeholder_with_default(False, shape=(), name='training')
my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training,
                              momentum=0.9)


with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1')
    bn1 = my_batch_norm_layer(hidden1)
    bn1_act = tf.nn.elu(bn1)

    hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2')
    bn2 = my_batch_norm_layer(hidden2)
    bn2_act = tf.nn.elu(bn2)

    logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name='outputs')
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

mnist = input_data.read_data_sets('/tmp/data')

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops],
                     feed_dict={training: True, X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print(epoch, 'Train accuracy: ', acc_train, 'Test accuracy:', acc_test)

    save_path = saver.save(sess, './my_model_final.ckpt')


[v.name for v in tf.trainable_variables()]
[v.name for v in tf.global_variables()]
