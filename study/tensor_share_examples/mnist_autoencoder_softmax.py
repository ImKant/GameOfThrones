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
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)



learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10
n_input = 784

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# 用字典的方式存储各隐藏层的参数
n_hidden_1 = 256 # 第一编码层神经元个数
n_hidden_2 = 128 # 第二编码层神经元个数
# 权重和偏置的变化在编码层和解码层顺序是相逆的
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

def auto_encoder(x, weights, biases):
    # 每一层结构都是 xW + b
    # 构建编码器
    with tf.name_scope("Encoder"):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))


    with tf.name_scope("Decoder"):
        # 构建解码器
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h1']),
                                       biases['decoder_b1']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h2']),
                                       biases['decoder_b2']))
    return layer_4

with tf.name_scope("AE"):
    # 预测
    y_encode = auto_encoder(X, weights, biases)
    y_true = X

    # 定义代价函数和优化器
    cost = tf.reduce_mean(tf.pow(y_true - y_encode, 2)) #最小二乘法
    ae_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)








# Parameters
learning_rate = 0.03
training_epochs = 25
batch_size = 100
display_step = 100
logs_path = '/tmp/tensorflow_logs/example'

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')

x2 = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')


# Create model
def multilayer_perceptron(x,x2, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['w1'])+tf.matmul(x2, weights['w2']), biases['b1'])
    out_layer = tf.nn.softmax(layer_1)
    tf.histogram_summary("sigmoid", layer_1)
    return out_layer

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_classes],stddev=0.3), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_input, n_classes],stddev=0.3), name='W2')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_classes],stddev=0.3), name='b1')
}

# Encapsulating all ops into scopes, making Tensorboard's Graph
# Visualization more convenient
with tf.name_scope('Model'):
    # Build model
    pred = multilayer_perceptron(x,x2, weights, biases)

with tf.name_scope('Loss'):
    # Softmax Cross entropy (cost function)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

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

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.scalar_summary("loss", loss)
# Create a summary to monitor accuracy tensor
tf.scalar_summary("accuracy", acc)
# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.histogram_summary(var.name, var)
# Summarize all gradients
# for grad, var in grads:
#     tf.histogram_summary(var.name + '/gradient', grad)
# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.train.SummaryWriter(logs_path,
                                            graph=tf.get_default_graph())


    ################### auto encoder training ########################

    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
    total_batch = int(mnist.train.num_examples/batch_size) #总批数

    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([ae_optimizer, cost], feed_dict={X: batch_xs})
        if i % display_step == 0:
            print("Epoch:", '%04d' % (i), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")





    # Training cycle
    avg_cost = 0.
    # Loop over all batches
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs_encode = sess.run(y_encode, feed_dict={X:batch_xs})
        # Run optimization op (backprop), cost op (to get loss value)
        # and summary nodes
        _, l, summary, accuracy = sess.run([apply_grads, loss, merged_summary_op, acc],
                                           feed_dict={x: batch_xs_encode, x2:batch_xs, y: batch_ys})
        # Write logs at every iteration
        summary_writer.add_summary(summary,  i)
        # Compute average loss
        # Display logs per epoch step
        if i % display_step == 0:
            print("Epoch:", '%04d' % i, "loss={:.9f}".format(l), "acc={:.9f}".format(accuracy))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    batch_xs_encode = sess.run(y_encode, feed_dict={X:mnist.test.images})
    print("Accuracy:", acc.eval({x: batch_xs_encode, y: mnist.test.labels}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")