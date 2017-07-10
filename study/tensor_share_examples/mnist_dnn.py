# -*- coding: utf-8 -*-
'''''
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.02
dropout_rate = 0.6
training_epochs = 100
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example'

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')


# Create model
# 第0层未输入层
# 最后一层是全链接softmax输出层
def dnn_struct(x, depth, layers_size, dropout_rate):
    weights = {}
    biases = {}
    layers = {}
    layers[0] = x
    for d in range(1, depth-1):
        weights[d] = tf.Variable(tf.random_normal([layers_size[d-1], layers_size[d]], stddev=0.3), name='W'+str(d));
        biases [d] = tf.Variable(tf.random_normal([layers_size[d]], stddev=0.3), name='B'+str(d));
        layers[d]  = tf.add(tf.matmul(layers[d-1], weights[d]), biases[d])
        layers[d]  = tf.nn.dropout(layers[d], dropout_rate)
        layers[d]  = tf.nn.relu(layers[d])
        tf.summary.histogram("relu"+str(d), layers[d])

    # Output layer
    d = depth - 1
    weights[d] = tf.Variable(tf.random_normal([layers_size[d-1], layers_size[d]]), name='W'+str(d));
    biases [d] = tf.Variable(tf.random_normal([layers_size[d]]), name='B'+str(d));
    out_layer =  tf.nn.softmax( tf.add(tf.matmul(layers[d-1], weights[d]), biases[d]))
    return out_layer


# Encapsulating all ops into scopes, making Tensorboard's Graph
# Visualization more convenient
with tf.name_scope('Model'):
    # Build model
    pred = dnn_struct(x, 6, [784,128,128,128,128,10], dropout_rate)

with tf.name_scope('Loss'):
    # Softmax Cross entropy (cost function)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Op to calculate every variable gradient
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Op to update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# Summarize all gradients
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())

    total_batch = int(mnist.train.num_examples/batch_size)
    print("total_batch", total_batch)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        current_accuracy = 0.

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, a, summary = sess.run([apply_grads, loss, acc,  merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
            current_accuracy += a / total_batch
            # Display logs per epoch step
        # if (epoch+1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost), "accuracy =", "{:.9f}".format(current_accuracy))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")