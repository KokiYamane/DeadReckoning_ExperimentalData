import numpy as np

import NeuralNetwork


# データ読み込み関数
def loaddata(filename):
    data = np.loadtxt(filename, delimiter=',',
                      skiprows=1, unpack=False, dtype=str)
    speed = []
    xyzwave = []
    for row in data:
        speed.append(row[1].astype('f8'))
        xyzwave.append(row[2: 2 + 3 * 50].astype('f8'))
    speed = np.array(speed)
    x = np.array(xyzwave)
    t = speed[:, np.newaxis]
    return x, t


# データの読み込み
filename = '0925'
x, t = loaddata('data/ML/' + filename + '.csv')

# ニューラルネットワーク構築
shape = [3 * 50, 30, 10, 1]
NN = NeuralNetwork.NeuralNetwork(shape, batchNorm=True, activation='tanh',
                                 loss='MSE', dropoutRatio=0)

# 学習
NN.learn(x, t, epoch=10000, learningRate=0.001, batchSize=100,
         optimizer='Adam', graph=True)

# 重みファイル出力
NN.output(directory='param')

# テスト
x, t = loaddata('data/ML/0925.csv')
NN.test(x, t)
x, t = loaddata('data/ML/0912_1815.csv')
NN.test(x, t)
