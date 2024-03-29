"""
用来练习调优
tensorboard 数据可视化
模型优化：
1. activation : relu, sigmoid, tanh
2. weight initializer : he, xavier, normal, truncated_normal
3. optimizer : Adam, Momentum, Gradient, Descent
"""
import tensorflow as tf
import os
import pickle
import numpy as np

CIFAR_DIR = "./cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


# tensorflow.Dataset.
class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        # 归一化
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """return batch_size examples as a batch."""
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
test_data = CifarData(test_filenames, False)

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])
# [None], eg: [0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32*32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])


def convnet(inputs, activation, kernel_initializer):
    """封装，方便调参"""
    # conv1: 神经元图， feature_map, 输出图像
    conv1_1 = tf.layers.conv2d(inputs,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer=kernel_initializer,
                               name = 'conv1_1')
    conv1_2 = tf.layers.conv2d(conv1_1,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer=kernel_initializer,
                               name = 'conv1_2')

    # 16 * 16
    pooling1 = tf.layers.max_pooling2d(conv1_2,
                                       (2, 2), # kernel size
                                       (2, 2), # stride
                                       name = 'pool1')


    conv2_1 = tf.layers.conv2d(pooling1,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer=kernel_initializer,
                               name = 'conv2_1')
    conv2_2 = tf.layers.conv2d(conv2_1,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer=kernel_initializer,
                               name = 'conv2_2')
    # 8 * 8
    pooling2 = tf.layers.max_pooling2d(conv2_2,
                                       (2, 2), # kernel size
                                       (2, 2), # stride
                                       name = 'pool2')

    conv3_1 = tf.layers.conv2d(pooling2,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer=kernel_initializer,
                               name = 'conv3_1')
    conv3_2 = tf.layers.conv2d(conv3_1,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer=kernel_initializer,
                               name = 'conv3_2')
    # 4 * 4 * 32
    pooling3 = tf.layers.max_pooling2d(conv3_2,
                                       (2, 2), # kernel size
                                       (2, 2), # stride
                                       name = 'pool3')
    # [None, 4 * 4 * 32]
    flatten = tf.layers.flatten(pooling3)
    return flatten


# sigmoid: 53.39% vs relu: 73.35% on 10k train
# 默认tf.glorot_uniform_initializer : 76.53% 100k train
# tf.truncated_normal_initializer : 78.04% 100k train
# tf.keras.initializers,he_normal : 71.52% 100k train
flatten = convnet(x_image, tf.nn.relu, tf.truncated_normal_initializer(stddev=0.02))


y_ = tf.layers.dense(flatten, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
# y_ -> sofmax
# y -> one_hot
# loss = ylogy_

# indices
predict = tf.argmax(y_, 1)
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# 优化器调优
with tf.name_scope('train_op'):
    # train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    # GradientDescent : 12.35% train 100k
    # train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
    # Momentum : 35.75% train 100k
    # 比较差的原因：1. initializer incorrect，2. 不充分的训练
    train_op = tf.train.MomentumOptimizer(
        learning_rate=1e-4, momentum=0.9).minimize(loss)
"""
tensorboard的使用
1. 指定面板图上显示的变量
2. 训练过程中将这些变量计算出来，并输出到文件中
3. 文件解析： ./tensorboard --logdir=dir
"""


def variable_summary(var, name):
    """Constructs summary for statistics of a variable"""
    with tf.name_scope(name):
        # 平均值
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            # 平方差
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('histogram', var)


# with tf.name_scope('summary'):
#     variable_summary(conv1_1, 'conv1_1')
#     variable_summary(conv1_2, 'conv1_2')
#     variable_summary(conv2_1, 'conv2_1')
#     variable_summary(conv2_2, 'conv2_2')
#     variable_summary(conv3_1, 'conv3_1')
#     variable_summary(conv3_2, 'conv3_2')

# 指定显示变量
# 'loss' : <10, 1.1>, <20, 1.08>
loss_summary = tf.summary.scalar('loss', loss)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

source_image = (x_image + 1) * 127.5
inputs_summary = tf.summary.image('inputs_summary', source_image)

merged_summary = tf.summary.merge_all()
merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])
# 指定显示文件夹
LOG_DIR = '.'
run_label = 'run_vgg_tensorboard'
run_dir = os.path.join(LOG_DIR, run_label)
if not os.path.exists(run_dir):
    os.mkdir(run_dir)
train_log_dir = os.path.join(run_dir, 'train')
test_log_dir = os.path.join(run_dir, 'test')
if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

# 用来控制计算图的频率
output_summary_every_steps = 100

# train 10k: 73.4%
with tf.Session() as sess:
    sess.run(init)
    # 添加句柄
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_log_dir)
    # test部分
    fixed_test_batch_data, fixed_test_batch_labels = test_data.next_batch(batch_size)

    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)

        eval_ops = [loss, accuracy, train_op]
        should_output_summary = ((i+1) % output_summary_every_steps == 0)
        if should_output_summary:
            eval_ops.append(merged_summary)

        eval_ops_results = sess.run(
            eval_ops,
            feed_dict={
                x: batch_data,
                y: batch_labels})
        loss_val, acc_val = eval_ops_results[0:2]
        if should_output_summary:
            train_summary_str = eval_ops_results[-1]
            train_writer.add_summary(train_summary_str, i+1)
            test_summary_str = sess.run([merged_summary_test],
                                        feed_dict={
                                            x:fixed_test_batch_data,
                                            y:fixed_test_batch_labels
                                        })[0]
            test_writer.add_summary(test_summary_str, i+1)

        if (i + 1) % 100 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f'
                  % (i + 1, loss_val, acc_val))
        if (i + 1) % 1000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels \
                    = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict={
                        x: test_batch_data,
                        y: test_batch_labels
                    })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i + 1, test_acc))


