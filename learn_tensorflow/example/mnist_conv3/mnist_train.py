# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 配置网络参数
batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 0.0001
training_steps = 30000
moving_average_decay = 0.99
model_save_path = "./model/"
model_name = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32,
                       [None, mnist_inference.input_node],
                       name='x-input')
    y_ = tf.placeholder(tf.float32,
                        [None, mnist_inference.output_node],
                        name='y-input')
    # 正则化
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = mnist_inference.inference(x, regularizer)
    # 存储训练轮数变量，不可训练参数
    global_step = tf.Variable(0, trainable=False)
    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    # 在可训练的变量上使用滑动平均
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())

    # 计算损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(learning_rate_base,
                                               global_step,
                                               mnist.train.num_examples / batch_size,
                                               learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
                    .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始化tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 验证和测试独立完成
        for i in range(training_steps):
            xs, ys = mnist.train.next_batch(batch_size)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))
                # 保存模型
                saver.save(sess,
                           os.path.join(model_save_path, model_name),
                           global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()



