import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import shutil
import time


class NewralNetwork:
    def __init__(self, inputData, teacherData, shape=[1, 3, 1], activation='sigmoid'):
        self.inputData = inputData          # 入力データ
        self.teacherData = teacherData      # 教師データ
        self.outputData = None              # 出力
        self.lossList = []                  # 損失関数

        # ニューラルネットワーク
        self.activation = activation
        self.layers = []
        for i in range(len(shape) - 1):
            self.layers.append(self.Affine(shape[i], shape[i+1]))
            if activation == 'relu': self.layers.append(self.Relu())
            else:                    self.layers.append(self.Sigmoid())
        self.layers.append(self.MSE())

        # グラフ
        self.fig = plt.figure(figsize=(10, 5))
        self.lossGraph = self.fig.add_subplot(2, 2, 1)
        self.outputGraph = self.fig.add_subplot(2, 2, 2)
        self.activationGraphs = []
        hiddenLayernum = len(shape) - 2
        for i in range(hiddenLayernum):
            self.activationGraphs.append(self.fig.add_subplot(
                2, hiddenLayernum, hiddenLayernum + 1 + i))
        plt.subplots_adjust(wspace=0.3, hspace=0.5)

    def learn(self, epoch=1000, batchSize=100, optimizer='Adam', learningRate=0.1, momentum=0.9, beta1=0.9, beta2=0.999, graph=False):
        start = time.time()
        for i in range(epoch):
            # 準伝播
            if batchSize > len(self.inputData): batchSize = len(self.inputData)
            batchMask = np.random.choice(len(self.inputData), batchSize)
            y = self.inputData[batchMask]
            for j in range(len(self.layers) - 1):
                y = self.layers[j].forward(y)
                if j == len(self.layers) - 2: self.outputData = y
            loss = self.layers[-1].forward(y, self.teacherData[batchMask])
            self.lossList.append(loss)

            # 逆伝播
            dy = None
            for j in reversed(range(len(self.layers))):
                if j == len(self.layers) - 1: dy = self.layers[j].backward()
                else:                         dy = self.layers[j].backward(dy)

            # 重み更新
            for j in range(0, len(self.layers) - 1, 2):
                dy = self.layers[j].update(
                    optimizer, learningRate, momentum, beta1, beta2)

            # グラフ表示
            if graph == True:
                if (i+1) % 100 == 0:
                    self.show(epoch, batchMask)

        # 経過時間表示
        elapsedTime = time.time() - start
        print('learning time  = {0:9.2f} [sec]'.format(elapsedTime))

        # 平均誤差表示
        learningErrer = np.average(self.outputData - self.teacherData[batchMask]) * 5
        print('learning errer = {0:9.6f} [m/s]'.format(learningErrer))

        # グラフ表示
        self.show(epoch, batchMask)
        plt.show()

    def show(self, epoch, batchMask):
        self.lossGraph.cla()
        self.lossGraph.plot(self.lossList)
        self.lossGraph.set_xlabel('iterations')
        self.lossGraph.set_ylabel('loss')
        self.lossGraph.set_xlim(-epoch * 0.05, epoch * 1.05)
        self.lossGraph.set_ylim(-max(self.lossList) * 0.05, max(self.lossList) * 1.05)
        self.lossGraph.grid()

        self.outputGraph.cla()
        self.outputGraph.scatter(range(len(self.inputData)), self.teacherData, label='teacher', color='gray')
        self.outputGraph.scatter(batchMask, self.outputData, label='output', marker='.')
        self.outputGraph.set_xlabel('datanum')
        self.outputGraph.set_ylabel('output')
        self.outputGraph.grid()
        self.outputGraph.legend()

        for i in range(len(self.activationGraphs)):
            self.activationGraphs[i].cla()
            data = np.reshape(self.layers[2*i+1].y, [-1, 1])
            self.activationGraphs[i].hist(data, 30, range=(0, 1))
            self.activationGraphs[i].set_title('{}-layer'.format(i+1))
            self.activationGraphs[i].grid()

        plt.draw()
        plt.pause(1e-10)

    def test(self, inputData, teacherData):
        y = inputData
        for j in range(len(self.layers)-1):
            y = self.layers[j].forward(y)
            if j == len(self.layers)-2:
                outputData = y

        plt.figure()
        plt.title('test')
        plt.plot(teacherData, label='theacher')
        plt.plot(outputData, label='output')
        plt.legend()
        plt.show()

        # 平均誤差表示
        testErrer = np.average(outputData - teacherData) * 5
        print('test errer     = {0:9.6f} [m/s]'.format(testErrer))

        return outputData

    def output(self, directory='param'):
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
        for i in range(0, len(self.layers)-1, 2):
            self.layers[i].output(directory, i)

    class Affine:
        def __init__(self, imputnum, outputnum):
            # self.w = np.random.randn(imputnum, outputnum) * np.sqrt(1/outputnum)
            # self.b = np.random.randn(1, outputnum)        * np.sqrt(1/outputnum)
            self.w = np.random.randn(imputnum, outputnum) * np.sqrt(2/outputnum)
            self.b = np.random.randn(1, outputnum)        * np.sqrt(2/outputnum)
            self.x = None
            self.dw = None
            self.db = None

            # momentum
            self.vw = np.zeros_like(self.w)
            self.vb = np.zeros_like(self.b)

            # AdaGrad
            self.hw = np.zeros_like(self.w)
            self.hb = np.zeros_like(self.b)

            # Adam
            self.mw = np.zeros_like(self.w)
            self.mb = np.zeros_like(self.b)
            self.vw = np.zeros_like(self.w)
            self.vb = np.zeros_like(self.b)

        def forward(self, x):
            self.x = x
            return np.dot(x, self.w) + self.b

        def backward(self, dy):
            self.dw = np.dot(self.x.T, dy)
            self.db = np.sum(dy, axis=0)
            return np.dot(dy, self.w.T)

        def update(self, optimizer='SGD', learningRate=0.1, momentum=0.9, beta1=0.9, beta2=0.999):
            if optimizer == 'momentum':
                self.vw = momentum * self.vw - learningRate * self.dw
                self.vb = momentum * self.vb - learningRate * self.db
                self.w += self.vw
                self.b += self.vb

            elif optimizer == 'AdaGrad':
                self.hw += self.dw**2
                self.hb += self.db**2
                self.w -= learningRate / self.hw**0.5 * self.dw
                self.b -= learningRate / self.hb**0.5 * self.db

            elif optimizer == 'Adam':
                self.mw += (1 - beta1) * (self.dw - self.mw)
                self.mb += (1 - beta1) * (self.db - self.mb)
                self.vw += (1 - beta2) * (self.dw**2 - self.vw)
                self.vb += (1 - beta2) * (self.db**2 - self.vb)
                self.w -= learningRate * self.mw / (np.sqrt(self.vw) + 1e-7)
                self.b -= learningRate * self.mb / (np.sqrt(self.vb) + 1e-7)

            else:
                self.w -= learningRate * self.dw
                self.b -= learningRate * self.db

        def output(self, directory='', i=0):
            np.savetxt(directory + '/w' + str(int(i / 2 + 1)) +
                       '.csv', self.w, delimiter=',')
            np.savetxt(directory + '/b' + str(int(i / 2 + 1)) +
                       '.csv', self.b, delimiter=',')

    class Sigmoid:
        def __init__(self):
            self.y = None

        def forward(self, x):
            self.y = 1 / (1 + np.exp(-x))
            return self.y

        def backward(self, dy):
            ds = self.y * (1.0 - self.y)
            return ds * dy

    class Relu:
        def __init__(self):
            self.mask = None
            self.y = None

        def forward(self, x):
            self.mask = (x <= 0)
            y = x.copy()
            y[self.mask] = 0
            self.y = y
            return y

        def backward(self, dy):
            dx = dy.copy()
            dx[self.mask] = 0
            return dx

    class MSE:
        def __init__(self):
            self.t = None
            self.x = None

        def forward(self, x, t):
            self.x = x
            self.t = t
            return 0.5 * np.average((x - self.t)**2)

        def backward(self):
            return self.x - self.t


if __name__ == '__main__':
    # 学習データ
    x = np.arange(0, 6*np.pi, 0.001)
    x = x[:, np.newaxis]
    t = 0.5 * np.sin(x) + 0.5

    # ニューラルネットワーク構築
    shape = [1, 3, 3, 3, 1]
    NN = NewralNetwork(x, t, shape, activation='sigmoid')

    # 学習
    NN.learn(epoch=2000, batchSize=1000, optimizer='Adam', learningRate=0.01, graph=True)

    # 重みファイル出力
    NN.output()
