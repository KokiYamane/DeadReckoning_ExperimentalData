import numpy as np

import NeuralNetwork as NN


# データ読み込み関数
def loaddata(filename):
    data = np.loadtxt(filename, delimiter=',',
                      skiprows=1, unpack=False, dtype=str)
    speed = []
    xyzwave = []
    for row in data:
        speed.append(row[1].astype('f8'))
        xyzwave.append(row[2: 152].astype('f8'))
    speed = np.array(speed)
    teacherData = speed[:, np.newaxis]
    inputData = np.array(xyzwave)
    return inputData, teacherData


# データの読み込み
filename = '0912_1815'
x, t = loaddata('data/ML/' + filename + '.csv')

# ニューラルネットワーク構築
shape = [150, 30, 10, 1]
NN = NN.NewralNetwork(shape, batchNorm=True, activation='tanh')

# 学習
NN.learn(x, t, epoch=2000, learningRate=0.01, batchSize=100,
         optimizer='SGD', graph=True)

# 重みファイル出力
NN.output(directory='param')

# テスト
filename = '0912_1800'
inputData, teacherData = loaddata('data/ML/' + filename + '.csv')
NN.test(inputData, teacherData)
