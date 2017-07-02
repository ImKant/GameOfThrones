# 例1，生成三维数据，然后用一个平面拟合它：
# (tensorflow)$ python   用 Python API 写 TensorFlow 示例代码

import tensorflow as tf
import numpy as np

# 用 NumPy 随机生成 100 个数据
#x_data = np.float32(np.random.rand(2, 100))
#y_data = np.dot([-0.100, 0.200], x_data) + 0.300

xph=tf.placeholder(tf.float32, [2, 100])
yph=tf.placeholder(tf.float32, 100)

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, xph) + b

# 最小化方差
loss = tf.reduce_mean(tf.square (y - yph))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([-0.100, 0.200], x_data) + 0.300
# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
with tf.Session() as sess:
    sess.run(init)

    # 拟合平面
    print sess.run([train,loss], feed_dict={xph:x_data, yph:y_data})
    for step in xrange(0, 1000):
        result = sess.run([train,loss], feed_dict={xph:x_data, yph:y_data})
        if step % 100 == 0:
            print step, sess.run(W), sess.run(b), result
            pass

    print sess.run([W,b])