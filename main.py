import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from flask import Flask,jsonify,render_template,request
import json
from mnist import model

x = tf.placeholder('float', [None, 784])

sess = tf.Session()

with tf.variable_scope('regression'):
    y1, variables1 = model.regression(x)
    variables = variables1
saver = tf.train.Saver(variables)
saver.restore(sess,"mnist/data/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder('float')
    y2, variables2 = model.convolutional(x,keep_prob)
    variables = list(set(variables1) ^ set(variables2))
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")

with tf.variable_scope("cnn_lstm"):
    y3, variables3 = model.cnn_lstm(x)
    variables = list(set(variables2) ^ set(variables3))
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/cnn_lstm.ckpt")

def regression(input):
    return sess.run(y1, feed_dict={x:input}).flatten().tolist()
def convolutional(input):
    return sess.run(y2, feed_dict={x:input,keep_prob:1.0}).flatten().tolist()
def cnn_lstm(input):
    return sess.run(y3, feed_dict={x:input}).flatten().tolist()


app = Flask(__name__)
@app.route('/')
def main():
    return render_template('index.html')
@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1,784)
    output1 = regression(input)
    output2 = convolutional(input)
    output3 = cnn_lstm(input)

    output = {}
    output["output1"] = output1
    output["output2"] = output2
    output["output3"] = output3
    res = []
    res.append(output)
    a = {}
    a['site'] = res
    mydata = json.dumps(a, ensure_ascii=False).encode("utf8")
    return mydata



if __name__ == "__main__":
    app.debug = True
    app.run(port=9000)