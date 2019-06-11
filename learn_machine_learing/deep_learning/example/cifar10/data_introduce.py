# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
import tensorflow as tf

# 导入数据
CIFAR_DIR = "./cifar-10-batches-py"

with open(os.path.join(CIFAR_DIR, "data_batch_1"), 'rb') as f:
    data = pickle.load(f,encoding='bytes')


def load_data(filename):
    """read data from data file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


# 构建图
x = tf.placeholder(tf.float32, [None, 3027])
y = tf.placeholder(tf.int64, [None])
w = tf.get_variable('w', [x.get_shape()[-1], 1],
                    initializer=tf.random_normal_initalizer(0, 1))
b = tf.get_variable('b', [1],
                    initializer=tf.constant_initializer(0,0))

y_ = tf.matmul(x, w) + b
p_y_1 = tf.nn.sigmoid(y_)

y_reshaped = tf.reshape(y, (-1, 1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))

predict = p_y_1>0.5
correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshaped)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))











