# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
learning_rate = 0.01
n_input = 784
batch_size = 50
training_epochs = 10

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)     #下载并加载mnist数据
x = tf.placeholder(tf.float32, [None, n_input])                        #输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 10])            #输入的标签占位符

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

# 每一层结构都是 xW + b
# 构建编码器
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# 构建解码器
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# 构建模型
with tf.name_scope('AUTOENCODER'):
    encoder_op = encoder(x)
    decoder_op = decoder(encoder_op)

    # 预测
    y_pred = decoder_op
    y_true = x

    # 定义代价函数和优化器
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #最小二乘法
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape, name='w'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

#定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape,name='b'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

#定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#构建网络
with tf.name_scope('CNN'):
    with tf.name_scope('conv_layer1'):
        x_image = tf.reshape(x, [-1,28,28,1])         #转换输入数据shape,以便于用于网络中
        W_conv1 = weight_variable([5, 5, 1, 32], name="W_conv1")
        b_conv1 = bias_variable([32], name="b_conv1")
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     #第一个卷积层
        h_pool1 = max_pool(h_conv1)                                  #第一个池化层

    with tf.name_scope('conv_layer2'):
        W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")
        b_conv2 = bias_variable([64], name="b_conv2")
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #第二个卷积层
        h_pool2 = max_pool(h_conv2)                                   #第二个池化层

    with tf.name_scope('full_connect_layer'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W_fc1")
        b_fc1 = bias_variable([1024], name="b_fc1")
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])              #reshape成向量
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #第一个全连接层

    with tf.name_scope('dropout_layer'):
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout层

    with tf.name_scope('softmax_layer'):
        W_fc2 = weight_variable([1024, 10], name="W_fc2")
        b_fc2 = bias_variable([10], name="b_fc2")

    with tf.name_scope('wide_layer'):
        W_wide = weight_variable([784, 10], name="W_wide")


    with tf.name_scope('output'):
        y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + tf.matmul(x, W_wide) + b_fc2)   #softmax层

    with tf.name_scope('sgd'):
        cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     #交叉熵
        train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)    #梯度下降法

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 #精确度计算
        # tf.summary.scalar("acc", accuracy)
saver = tf.train.Saver()

# for var in tf.trainable_variables():
#     tf.summary.histogram(var.name, var)

tf.summary.scalar("loss", cross_entropy)


merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('/tmp/mnist_cnn_ultimate_logs', graph=tf.get_default_graph())

    sess.run(tf.initialize_all_variables())

    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
        if i % 100 == 0:
            print("Epoch:", '%04d' % (i), "cost=", "{:.9f}".format(c))
    print("AutoEncoder Optimization Finished!")


    for i in range(100000):
        batch = mnist.train.next_batch(50)
        x_encode = sess.run(y_pred, feed_dict={x: batch[0]})

        if i%100 == 0:                  #训练100次，验证一次
            train_acc = accuracy.eval(feed_dict={x:x_encode, y_actual: batch[1], keep_prob: 1.0})
            print('step',i,'training accuracy',train_acc)
            # tf.runable_variable()
            c,summary_str = sess.run([train_step,merged_summary_op], feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

            summary_writer.add_summary(summary_str, i)

    x_encode = sess.run(y_pred, feed_dict={x: mnist.test.images[0:1000]})

    test_acc=accuracy.eval(feed_dict={x: x_encode, y_actual: mnist.test.labels[0:1000], keep_prob: 1.0})
    print("test accuracy encoded",test_acc)

    test_acc=accuracy.eval(feed_dict={x:  mnist.test.images[0:1000], y_actual: mnist.test.labels[0:1000], keep_prob: 1.0})
    print("test accuracy",test_acc)

    save_path = saver.save(sess, "tmp/model.ckpt")
    print "Model saved in file: ", save_path