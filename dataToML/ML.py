import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import time
import NeuralNetwork as NN

# データの読み込み
filename = '0912_1815'
data = np.loadtxt(
        'MLdata/' + filename + '.txt',
        delimiter=',', skiprows=1, unpack=False, dtype=str)
speed = []
xwave = []
ywave = []
zwave = []
xyzwave = []
wavedatanum = 50
for row in data:
    speed.append(row[1].astype('float64'))
    xwave.append(row[2 : 2+wavedatanum].astype('float64'))
    ywave.append(row[2+wavedatanum : 2+wavedatanum*2].astype('float64'))
    zwave.append(row[2+wavedatanum*2 : 2+wavedatanum*3].astype('float64'))
    xyzwave.append(row[2 : 2+wavedatanum*3].astype('float64'))
speed = np.array(speed)
xwave = np.array(xwave)
ywave = np.array(ywave)
zwave = np.array(zwave)
xyzwave = np.array(xyzwave)

# ニューラルネットワーク構築
teach = speed[:, np.newaxis]
shape = [150, 10, 1]
NN = NN.NewralNetwork(xyzwave, teach, shape, 10000, 0.1)

# 学習
NN.learn()

# 重みファイル出力
NN.output()
