import os
import model
import tensorflow as tf
import input_data

data = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义模型
with tf.variable_scope('cnn_lstm'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y, variables = model.cnn_lstm(x)

# 训练
y_ = tf.placeholder('float', [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver(variables)

with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summary_write = tf.summary.FileWriter('tmp/mnist_log/1', sess.graph)
    summary_write.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print("Step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    result = []
    for i in range(2000):
        batch = data.test.next_batch(50)
        result.append(sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]}))
    print(sum(result)/len(result))

    path = saver.save(
        sess,
        os.path.join(os.path.dirname(__file__), 'data', 'cnn_lstm.ckpt'),
        write_meta_graph=False,
        write_state=False
    )
    print("Saved:", path)
