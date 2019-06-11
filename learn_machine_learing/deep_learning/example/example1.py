'''
这是一个深度学习的例子。
解决的是，典型的二分类问题。
实现：将电影影评划分正负面。
'''
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# 加载IMDB数据集
(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words = 10000)

# 解码评论
# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

# 准备数据


def vectorize_sequeces(sequeces, dimension=10000):
    results = np.zeros((len(sequeces), dimension))
    for i, sequece in enumerate(sequeces):
        results[i, sequece] = 1.
    return results


x_train = vectorize_sequeces(train_data)
x_test = vectorize_sequeces(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 构建网络
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 训练网络
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)

# 预测结果
model.predict(x_test)


# history_dict = history.history
# print( history_dict.keys() )
#
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
#
# loss_values = history_dict['loss']
# epochs = range(1, len(loss_values)+1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()



















