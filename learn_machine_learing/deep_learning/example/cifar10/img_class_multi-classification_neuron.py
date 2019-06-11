# -*- coding: utf-8 -*-
# 构建图的过程
# 单个神经元，多分类例子
import pickle
import numpy as np
import os
import tensorflow as tf

# 导入数据
CIFAR_DIR = "./cifar-10-batches-py"

def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']

# --------------处理数据-----------------
class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1 # 归一化
        self._labels = np.hstack(all_labels)

        print(self._data.shape)
        print(self._labels.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        """shuffle the data"""
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """return batch_size example as a batch"""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]
train_data = CifarData(train_filenames, True)
# test_data = CifarData(test_filenames, False)
# --------------处理数据-----------------
# --------------构建图-----------------
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])
# (3072, 10)
w = tf.get_variable('w', [x.get_shape()[-1], 10],
                    initializer=tf.random_normal_initializer(0, 1))
# (10, )
b = tf.get_variable('b', [10],
                    initializer=tf.constant_initializer(0.0))
# [None, 3072] * [3072, 10] = [None, 10]
y_ = tf.matmul(x, w) + b

"""使用平方差计算，也可以使用底下的交叉熵方式
# api实现 : e^x / sum(e^x)
# 得到概率分布的数据，和为1
p_y = tf.nn.softmax(y_)
y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(y_one_hot - p_y))
"""
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
# 做了下面动作：y_ -> sofmax
# y -> one_hot
# loss = y log y_

"""二分类的做法
# [None, 1]  添加激活函数sigmoid
p_y_1 = tf.nn.sigmoid(y_)
# 需要和y做差别分析，但是类型不一样，需要转一下，并数据类型一样
y_reshaped = tf.reshape(y, (-1, 1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)
# loss就是均值
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))
"""
# indices
predict = tf.argmax(y_, 1)
# [1,0,1,1,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# 优化器
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# --------------构建图-----------------
# 执行图
init = tf.global_variables_initializer()
batch_size = 20
train_steps = 100000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ = sess.run(
            [loss, accuracy, train_op],
            feed_dict={
                x: batch_data,
                y: batch_labels
            })
        if (i+1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i, loss_val, acc_val))
        if (i+1) % 5000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict = {
                        x: test_batch_data,
                        y: test_batch_labels
                    })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Train] Step: %d, acc: %4.5f' % (i+1, test_acc))





