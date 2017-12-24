# coding:utf-8
'''
Created on 2017/12/24

@author: Dxq
'''
import tensorflow as tf

va = tf.constant(1)
vb = tf.Variable(2)
vc = tf.placeholder(dtype=tf.int32)
res = va + vb + vc
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    output = sess.run(res, feed_dict={vc: 3})
    print(output)
