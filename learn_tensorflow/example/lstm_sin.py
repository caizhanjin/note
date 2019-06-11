# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

hidden_size = 30
num_layers = 2

time_steps = 10
training_steps = 10000
batch_size = 32

training_examples = 10000
testing_examples = 1000
sample_gap = 0.01


def generate_data(seq):
    """用sin函数前面的time_steps个点的值，预测第i + time_steps点的函数值"""
    X = []
    y = []
    for i in range(len(seq) - time_steps):
        X.append([seq[i: i + time_steps]])
        y.append([seq[i + time_steps]])
    # x->[None, time_steps] y->[None,]
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y, is_training):
    """lstm_model 模型"""
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        for _ in range(num_layers)
    ])
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]
    predictions = tf.contrib.layers.fully_connected(
        output, 1, activation_fn=None
    )

    if not is_training:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # 优化
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.train.get_global_step(),
        optimizer="Adagrad",
        learning_rate=0.1
    )

    return predictions, loss, train_op


def train(sess, train_X, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(training_steps):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))


def run_eval(sess, test_X, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)
    predictions = []
    labels = []
    for i in range(testing_examples):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Mean Square Error is : %f" % rmse)

    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='labels')
    plt.legend()
    plt.show()


test_start = (training_examples + time_steps) * sample_gap
test_end = test_start + (testing_examples + time_steps) * sample_gap
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, training_examples+time_steps, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, testing_examples + time_steps, dtype=np.float32)))

with tf.Session() as sess:
    train(sess, train_X, train_y)
    run_eval(sess, test_X, test_y)













