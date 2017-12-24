# coding:utf-8
'''
Created on 2017/12/24

@author: Dxq
'''
import tensorflow as tf

va = tf.constant(1)
vb = 2
res = va+vb
print(res)
# Tensor("add:0", shape=(), dtype=int32)
with tf.Session() as sess:
    output = sess.run(res)
    print(output)
    # 3