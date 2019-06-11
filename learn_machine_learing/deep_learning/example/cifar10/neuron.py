# -*- coding: utf-8 -*-
# 构建图的过程
# 单个神经元
import pickle
import numpy as np
import os

# 导入数据
CIFAR_DIR = "./cifar-10-batches-py"

def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']

# 构建图
x = tf.placeholder(tf.float32, [None, 3027])
y = tf.placeholder(tf.int64, [None])
# (3072, 1)
w = tf.get_variable('w', [x.get_shape()[-1], 1],
                    initializer=tf.random_normal_initalizer(0, 1))
# (1, )
b = tf.get_variable('b', [1],
                    initializer=tf.constant_initializer(0,0))
# [None, 3072] * [3072, 1] = [None, 1]
y_ = tf.matmul(x, w) + b
# [None, 1]  添加激活函数sigmoid
p_y_1 = tf.nn.sigmoid(y_)
# 需要和y做差别分析，但是类型不一样，需要转一下，并数据类型一样
y_reshaped = tf.reshape(y, (-1, 1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)
# loss就是均值
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))

# bool
predict = p_y_1>0.5
# [1,0,1,1,0]
correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshaped)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# 优化器
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOpimizer(le-3).minimize(loss)








