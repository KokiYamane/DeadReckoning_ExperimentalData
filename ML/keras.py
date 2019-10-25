# 警告無視
import warnings
warnings.simplefilter('ignore', FutureWarning)

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# matplotlib 日本語フォント設定
from matplotlib import rcParams
rcParams['font.family'] = ['Yu Gothic', 'Corporate Logo Medium']

# バージョン表示
print('numpy version:', np.__version__)
print('tensorflow version:', tf.__version__)
print('GPU:', tf.test.gpu_device_name())

# データ読み込み関数
def loadData(filename):
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
def shuffleData(x, y):
    zipped = list(zip(x, y))
    np.random.shuffle(zipped)
    x_result, y_result = zip(*zipped)
    return np.asarray(x_result), np.asarray(y_result)

# 学習データとテストデータに分割
def splitData(x, t):
    border = int(len(x) * 0.9)
    return (x[:border], t[:border]), (x[border:], t[border:])

# データの読み込み
x, t = loadData('ML/data/ML/1010.csv')
x_shuffle, t_shuffle = shuffleData(x, t)
(x_train, t_train), (x_test,  t_test) = splitData(x_shuffle, t_shuffle)

# モデル生成
model = keras.Sequential()
model.add(keras.layers.Input(shape=(150,)))
model.add(keras.layers.Dense(100, activation='tanh'))
model.add(keras.layers.Dense( 70, activation='tanh'))
model.add(keras.layers.Dense( 40, activation='tanh'))
model.add(keras.layers.Dense( 10, activation='tanh'))
model.add(keras.layers.Dense(  1))
adam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse')

# モデルの訓練
result = model.fit(x_train, t_train, epochs=100)

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

# axes[1].set_title('train data')
axes[1].set_title('学習データ')
axes[1].set_xlabel('time [s]')
axes[1].set_ylabel('speed [m/s]')
axes[1].plot(t_train, color='gray', label='teacher')
axes[1].plot(y_train, label='output', alpha=0.8)
axes[1].plot(error_train, color='red', label='error')
axes[1].legend()
axes[1].grid()

# axes[2].set_title('test data')
axes[2].set_title('テストデータ')
axes[2].set_xlabel('time [s]')
axes[2].set_ylabel('speed [m/s]')
axes[2].plot(t_test, color='gray', label='teacher')
axes[2].plot(y_test, label='output')
axes[2].plot(error_test, color='red', label='error')
axes[2].legend()
axes[2].grid()

plt.subplots_adjust(hspace=0.5)
fig.savefig('ML/graph/graph.png')

# 誤差算出
error_train = np.abs(t_train - y_train)
error_test  = np.abs(t_test - y_test)

# 結果表示
print('error_train_average = {}'.format(np.average(error_train)))
print('error_train_max     = {}'.format(np.max(error_train)))
print('error_test_average  = {}'.format(np.average(error_test)))
print('error_test_max      = {}'.format(np.max(error_test)))

plt.show()
