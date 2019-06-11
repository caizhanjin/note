# -*- coding: utf-8 -*-
"""定义前向传播的过程和神经网络的参数"""
import tensorflow as tf

# 定义模型参数
input_node = 784
output_node = 10
layer1_node = 500


# 通过tf.get_variable来获取变量
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights",
                              shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 将当前变量正则损失加入“losses”集合，自定义，不在tensorflow自动管理集合列表中
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义前向传播过程
def inference(input_tensor, regularizer):
    # 第一层网络
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([input_node, layer1_node],
                                      regularizer)
        biases = tf.get_variable("biases", [layer1_node],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # 第二层网络
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([layer1_node, output_node],
                                      regularizer)
        biases = tf.get_variable("biases", [output_node],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
