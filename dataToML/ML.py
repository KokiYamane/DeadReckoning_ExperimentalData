import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import NeuralNetwork as NN

# データ読み込み関数
def loaddata(filename):
    data = np.loadtxt(filename, delimiter=',',
                      skiprows=1, unpack=False, dtype=str)
    speed = []
    xyzwave = []
    for row in data:
        speed.append(row[1].astype('float64'))
        xyzwave.append(row[2: 152].astype('float64'))
    speed = np.array(speed)
    teacherData = speed[:, np.newaxis]
    inputData = np.array(xyzwave)
    return inputData, teacherData


# データの読み込み
filename = '0912_1815'
inputData, teacherData = loaddata('MLdata/' + filename + '.txt')

# ニューラルネットワーク構築
shape = [150, 30, 10, 1]
NN = NN.NewralNetwork(inputData, teacherData, shape)

# 学習
NN.learn(epoch=2000, batchSize=1000, optimizer='Adam', learningRate=0.01, graph=True)

# 重みファイル出力
NN.output()

# テスト
filename = '0912_1800'
inputData, teacherData = loaddata('MLdata/' + filename + '.txt')
NN.test(inputData, teacherData)
