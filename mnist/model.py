import tensorflow as tf


# 定义线性模型 Y = W * X + b
def regression(x):
    W = tf.Variable(tf.zeros([784, 10]), name='W')
    b = tf.Variable(tf.zeros([10]), name='b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    train_vars = tf.trainable_variables()
    return y, train_vars


# 定义卷积模型
def convolutional(x, keep_prob):
    # 定义卷积层
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')

    # 定义池化层
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # full connection
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # full connection
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    train_vars = tf.trainable_variables()
    return y, train_vars


def cnn_lstm(x):
    # 以正太分布初始化weight
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 以0.1这个常量来初始化bias
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # 池化
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('input'):
        x = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope('conv_pool_1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('conv_pool_2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    X_in = tf.reshape(h_pool2, [-1, 49, 64])
    X_in = tf.transpose(X_in, [0, 2, 1])

    with tf.variable_scope('lstm'):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            128, forget_bias=1.0, state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(
            lstm_cell, X_in, time_major=False, dtype=tf.float32)
        W_lstm = weight_variable([128, 10])
        b_lstm = bias_variable([10])
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        y = tf.nn.softmax(tf.matmul(outputs[-1], W_lstm) + b_lstm)
    train_vars = tf.trainable_variables()
    return y, train_vars