import numpy as np

from mnist.mnist import load_mnist
import NeuralNetwork

# データの読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True, one_hot_label=True)

# ニューラルネットワーク構築
shape = [784, 100, 100, 100, 100, 100, 10]
NN = NeuralNetwork.NeuralNetwork(shape, batchNorm=True, activation='tanh',
                                 loss='CrossEntropy', dropoutRatio=0.2)

# 学習
NN.learn(x_train, t_train, epoch=3000, learningRate=0.01, batchSize=100,
         optimizer='Adam', graph=True)

# テスト
NN.test(x_test, t_test)

# 重みファイル出力
NN.output(directory='param')
