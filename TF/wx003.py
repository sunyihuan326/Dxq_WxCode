# coding:utf-8
'''
Created on 2017/12/24

@author: Dxq
'''
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt


def make_data():
    np.random.seed(1)
    trX = np.random.randn(100)
    noise = (20 * np.random.randn(100) - 10) / 20
    trY = trX * 2.0 + 1.0 + noise
    return trX, trY


def plot_res(X, Y, a, b):
    plt.figure(1)
    plt.scatter(X, Y, s=10, c='b')
    plt.plot(X, X * a + b, c='r')
    plt.savefig('1.png')


epoches = 20

X = tf.placeholder(dtype=tf.float32, shape=[None, ])
Y = tf.placeholder(dtype=tf.float32, shape=[None, ])

a = tf.Variable(1.0, dtype=tf.float32)
b = tf.Variable(0.0, dtype=tf.float32)

y_hat = tf.add(tf.multiply(a, X), b)
loss = tf.reduce_mean(tf.square(Y - y_hat))
optimizer_op = tf.train.GradientDescentOptimizer(.5).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    trX, trY = make_data()

    for epoch in range(epoches):
        pre_y, _loss, _ = sess.run([y_hat, loss, optimizer_op], feed_dict={X: trX, Y: trY})
        if epoch % 5 == 0:
            _a = a.eval()
            _b = b.eval()
            print('current step:{} | a:{} | b:{} | loss:{}'.format(epoch, _a, _b, _loss))

    plot_res(trX, trY, _a, _b)
