'''
这是深度学习的一个学习例子，涉及经典的回归问题。
实现：根据房地产数据估算房屋价格。
'''
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据标准化：数据的取值范围差异大，需要标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# 构建模型
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

######### 尝试阶段 #########
# K折验证
# k = 4
# num_val_samples = len(train_data) // k
# num_epochs = 500
# all_mae_histories = []
#
# for i in range(k):
#     print('processing fold #', i)
#     # 验证数据
#     val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
#     val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]
#     # 训练数据
#     partial_train_data = np.concatenate(
#         [train_data[ : i*num_val_samples],
#          train_data[(i+1)*num_val_samples : ]],
#         axis=0)
#     partial_train_targets = np.concatenate(
#         [train_targets[ : i*num_val_samples],
#          train_targets[(i+1)*num_val_samples : ]],
#         axis=0)
#
#     model = build_model()
#     history = model.fit(partial_train_data,
#                         partial_train_targets,
#                         validation_data=(val_data, val_targets),
#                         epochs=num_epochs,
#                         batch_size=1,
#                         verbose=0)
#
#     mae_history = history.history['val_mean_absolute_error']
#     all_mae_histories.append(mae_history)
#
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
#
# def smooth_curve(points, factor=0.9):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous*factor + point*(1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points
#
# smooth_mae_history = smooth_curve(average_mae_history[10 : ])
#
# plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()
# 从折线图中看出，38轮次时MEA最低，之后就过拟合
######### 尝试阶段 #########

# 训练最终模型
model = build_model()
model.fit(train_data,
          train_targets,
          epochs=100,
          batch_size=16,
          verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mae_score)









