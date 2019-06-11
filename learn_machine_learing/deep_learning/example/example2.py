'''
这是深度学习的一个例子。
解决的是典型的多分类问题。
实现：将新闻按主题分类。
'''
from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt
# 下载数据
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# 有兴趣可以解码索引为新闻文本
# word_index = reuters.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_newswire = ' '.join([reverse_word_index.get(i-1, '?') for i in train_data[0]])

# 准备数据，数据向量化


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
# def to_one_hot(labels, dimension=46):
#     results = np.zeros(len(labels), dimension)
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 标签向量化可以采用to_one_hot()，也可以用内置方法
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# 构建网络（模型）
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 第一遍，试验，后面优化
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss)+1)
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Traning and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# 优化，重新训练模型；9次后开始过拟合，所以轮次选择9次
model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 评价模型
reuters = model.evaluate(x_test, one_hot_test_labels)
print(reuters)

# 使用模型预测
predictions = model.predict(x_test)

print(predictions)
print(predictions[0])
print( predictions[0].shape )
print( np.sum(predictions[0]) )
# 取概率最大的为预测结果
print( np.argmax(predictions[0]) )

















