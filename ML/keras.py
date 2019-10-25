# 警告無視
import warnings
warnings.simplefilter('ignore', FutureWarning)

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

print('numpy version:', np.__version__)
print('tensorflow version:', tf.__version__)
print('GPU:', tf.test.gpu_device_name())

# データ読み込み関数
def loaddata(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1,
                      unpack=False, dtype=str)
    speed = []
    xyzwave = []
    for row in data:
        speed.append(row[1].astype('f8'))
        xyzwave.append(row[2: 2 + 3 * 50].astype('f8'))
    speed = np.array(speed)
    x = np.array(xyzwave)
    t = speed[:, np.newaxis]
    return x, t

# 入力・教師データを対応させたままシャッフル
def shuffle_samples(x, y):
    zipped = list(zip(x, y))
    np.random.shuffle(zipped)
    x_result, y_result = zip(*zipped)
    return np.asarray(x_result), np.asarray(y_result)

# データの読み込み
x_train1, t_train1 = loaddata('data/ML/0925.csv')
x_train2, t_train2 = loaddata('data/ML/1010.csv')
x_train = np.concatenate([x_train1, x_train2], axis=0)
t_train = np.concatenate([t_train1, t_train2], axis=0)
x_test, t_test = loaddata('data/ML/0912_1815.csv')

# x_train, t_train = loaddata('drive/My Drive/卒業研究/colab/data/ML/0925.csv')
# x_test,  t_test  = loaddata('drive/My Drive/卒業研究/colab/data/ML/1010.csv')

# シャッフル
x_train_shuffle, t_train_shuffle = shuffle_samples(x_train, t_train)

# モデル生成
model = keras.Sequential()
model.add(keras.layers.Dense(100, input_dim=150))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('tanh'))
# model.add(keras.layers.Dropout(0.2))

# model.add(keras.layers.Dense(100))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('tanh'))
# # model.add(keras.layers.Dropout(0.2))

# model.add(keras.layers.Dense(100))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('tanh'))
# # model.add(keras.layers.Dropout(0.2))

# model.add(keras.layers.Dense(100))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('tanh'))
# # model.add(keras.layers.Dropout(0.2))

# model.add(keras.layers.Dense(100))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('tanh'))
# # model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(1))

adam = keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=adam, loss='mse')

# モデルの可視化
# from IPython.display import SVG
# from tensorflow.python.keras.utils.vis_utils import model_to_dot
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='drive/My Drive/卒業研究/colab/graph/model.png',
#            show_shapes=True, show_layer_names=False)

# モデルの訓練
result = model.fit(x_train_shuffle, t_train_shuffle, epochs=100)

# 学習データでテスト
y_train = model.predict(x_train)
error_train = np.abs(t_train - y_train)

# テストデータでテスト
y_test = model.predict(x_test)
error_test  = np.abs(t_test - y_test)

# グラフ表示
fig, axes = plt.subplots(3, 1, figsize=(10,10), facecolor="white")

axes[0].set_title('LOSS')
axes[0].set_xlabel('epochs')
axes[0].set_ylabel('loss')
axes[0].plot(result.history['loss'], label="training")
axes[0].grid()

axes[1].set_title('train data')
axes[1].set_xlabel('time [s]')
axes[1].set_ylabel('speed [m/s]')
axes[1].plot(error_train, color='red', label='error')
axes[1].plot(y_train, label='output')
axes[1].plot(t_train, color='gray', label='teacher')
axes[1].legend()
axes[1].grid()

axes[2].set_title('test data')
axes[2].set_xlabel('time [s]')
axes[2].set_ylabel('speed [m/s]')
axes[2].plot(t_test, color='gray', label='teacher')
axes[2].plot(y_test, label='output')
axes[2].plot(error_test, color='red', label='error')
axes[2].legend()
axes[2].grid()

plt.subplots_adjust(hspace=0.5)
fig.savefig('graph/graph.png')

error_train = np.abs(t_train - y_train)
error_test  = np.abs(t_test - y_test)

print('error_train_average = {}'.format(np.average(error_train)))
print('error_train_max     = {}'.format(np.max(error_train)))
print('error_test_average  = {}'.format(np.average(error_test)))
print('error_test_max      = {}'.format(np.max(error_test)))

plt.show()
