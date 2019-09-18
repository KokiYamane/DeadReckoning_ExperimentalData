import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import NeuralNetwork as NN

# データの読み込み
def loaddata(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=False, dtype=str)
    speed = []
    xyzwave = []
    for row in data:
        speed.append(row[1].astype('float64'))
        xyzwave.append(row[2 : 152].astype('float64'))
    speed = np.array(speed)
    teacherdata = speed[:, np.newaxis]
    inputdata = np.array(xyzwave)
    return inputdata, teacherdata

# データの読み込み
filename = '0912_1815'
inputdata, teacherdata = loaddata('MLdata/' + filename + '.txt')

# ニューラルネットワーク構築
shape = [150, 30, 1]
NN = NN.NewralNetwork(inputdata, teacherdata, shape)

# 学習
NN.learn(method='Adam', epoch=2000, learningRate=0.01)

# 重みファイル出力
NN.output()

# テスト
filename = '0912_1800'
inputdata, teacherdata = loaddata('MLdata/' + filename + '.txt')
NN.test(inputdata, teacherdata)
