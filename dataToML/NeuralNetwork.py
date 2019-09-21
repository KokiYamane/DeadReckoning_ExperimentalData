import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np


class NewralNetwork:
    def __init__(self, shape=[1, 3, 1], batchNorm=False, activation='sigmoid'):
        self.x = None              # 入力
        self.t = None              # 教師データ
        self.y = None              # 出力
        self.loss = []             # 損失関数
        self.accuracy = []         # 正解率
        self.epoch = None          # 学習回数
        self.learningRate = None   # 学習率
        self.batchSize = None      # 一度に学習するデータ数

        # ニューラルネットワーク
        self.activation = activation
        self.layers = []
        for i in range(len(shape) - 1):
            self.layers.append(self.Layer(shape[i], shape[i + 1],
                                          batchNorm, activation))
        self.layers.append(self.MSE())

        # グラフ
        self.fig = plt.figure(figsize=(10, 6))
        self.lossGraph = self.fig.add_subplot(2, 2, 1)
        self.outputGraph = self.fig.add_subplot(2, 2, 2)
        self.activationGraphs = []
        # self.waightGraphs = []

        layerNum = len(shape) - 1
        self.activationGraphs.append(self.fig.add_subplot(
            2, layerNum, layerNum + 1))
        # self.waightGraphs.append(self.fig.add_subplot(
        #     3, layerNum, 2*layerNum + 1))
        for i in range(1, layerNum):
            self.activationGraphs.append(self.fig.add_subplot(
                2, layerNum, layerNum + 1 + i,
                sharex=self.activationGraphs[0],
                sharey=self.activationGraphs[0]))
            # self.waightGraphs.append(self.fig.add_subplot(
            #     3, layerNum, 2*layerNum + 1 + i))

        self.fig.align_labels()
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95,
                                 wspace=0.2, hspace=0.5)

    def learn(self, x, t, epoch=1000, learningRate=0.01, batchSize=100,
              graph=False, optimizer='SGD',
              momentum=0.9, beta1=0.9, beta2=0.999):
        if len(x) != len(t):
            print('deta size errer')
            exit()

        self.x = x
        self.t = t
        self.epoch = epoch
        self.learningRate = learningRate
        dataNum = len(x)
        if batchSize > dataNum:
            batchSize = dataNum
        self.batchSize = batchSize
    
        start = time.time()

        for i in range(epoch):
            # ミニバッチ生成
            batchMask = np.random.choice(dataNum, batchSize)
            y = x[batchMask]

            # 順伝播
            layerNum = len(self.layers)
            for j in range(layerNum - 1):
                y = self.layers[j].forward(y)
                if j == layerNum - 2:
                    self.y = y
            loss = self.layers[-1].forward(y, t[batchMask])

            # 順伝播の結果保存
            self.loss.append(loss)
            tBatch = t[batchMask]
            if len(x[0]) > 1:
                correction = []
                for j in range(len(y)):
                    correction.append(tBatch[j][np.argmax(y[j])])
                self.accuracy.append(np.average(correction) * 100)

            # 逆伝播
            dy = self.layers[layerNum - 1].backward()
            for j in reversed(range(layerNum - 1)):
                dy = self.layers[j].backward(dy)

            # 重み更新
            for j in range(layerNum - 1):
                self.layers[j].update(learningRate, optimizer,
                                      momentum, beta1, beta2)

            # グラフ表示
            if graph == True and (i+1) % 100 == 0:
                self.show(batchMask)

        # 経過時間表示
        elapsedTime = time.time() - start
        print('learning time = {0:7.2f} [sec]'.format(elapsedTime))

        if len(x[0]) ==  1:
            # 平均誤差表示
            learningErrer = np.average(np.abs(
                self.y - t[batchMask]))
            print('learning errer = {0:9.6f}'.format(learningErrer))
        else:
            # 正解率表示
            print('learning accuracy = {0:5.2f} [%]'.format(self.accuracy[-1]))

        # グラフ表示
        self.show(batchMask)
        plt.show()

    def show(self, batchMask):
        # 損失関数
        self.lossGraph.cla()
        self.lossGraph.plot(self.loss)
        self.lossGraph.set_xlabel('iterations')
        self.lossGraph.set_ylabel('loss')
        self.lossGraph.set_xlim(-self.epoch * 0.05, self.epoch * 1.05)
        ylim = max(self.loss)
        self.lossGraph.set_ylim(0, ylim * 1.05)
        self.lossGraph.grid()

        # 教師データと出力
        if len(self.x[0]) ==  1:
            self.outputGraph.cla()
            self.outputGraph.scatter(range(len(self.t)), self.t,
                                     label='teacher', color='gray')
            self.outputGraph.scatter(batchMask, self.y,
                                     label='output', marker='.')
            self.outputGraph.set_xlabel('datanum')
            self.outputGraph.set_ylabel('output')
            xlim = len(x)
            self.outputGraph.set_xlim(-xlim * 0.05, xlim * 1.05)
            self.outputGraph.grid()
            self.outputGraph.legend()

        # 認識精度
        else:
            self.outputGraph.cla()
            self.outputGraph.plot(self.accuracy)
            self.outputGraph.set_xlabel('iterations')
            self.outputGraph.set_ylabel('accuracy [%]')
            self.outputGraph.set_xlim(-self.epoch * 0.05, self.epoch * 1.05)
            self.outputGraph.set_ylim(0, 100)
            self.outputGraph.grid()

        # 隠れ層の出力
        for i in range(len(self.activationGraphs)):
            self.activationGraphs[i].cla()
            data = np.reshape(self.layers[i].activation.y, [-1, 1])
            weights = np.ones_like(data) / float(len(data))
            if self.activation == 'relu':
                histRange = None
            elif self.activation == 'tanh':
                histRange = (-1, 1)
            else:
                histRange = (0, 1)
            self.activationGraphs[i].hist(data, 20,
                                          weights=weights, range=histRange)
            self.activationGraphs[i].set_title('{}-layer'.format(i+1))
            self.activationGraphs[i].grid()
            if i != 0:
                self.activationGraphs[i].tick_params(
                    left=False, labelleft=False)
        self.activationGraphs[0].set_ylabel('activation')

        # 重み
        # for i in range(len(self.waightGraphs)):
        #     self.waightGraphs[i].cla()
        #     im = self.waightGraphs[i].imshow(
        #         self.layers[i].affine.w.T, cmap='gray')
        #     self.waightGraphs[i].tick_params(left=False, labelleft=False,
        #                                      bottom=False, labelbottom=False)
        # self.waightGraphs[0].set_ylabel('waight')
        
        # 表示
        plt.draw()
        plt.pause(1e-7)

    def test(self, x, t):
        # 順伝播
        layerNum = len(self.layers)
        y = x
        for j in range(layerNum - 1):
            y = self.layers[j].forward(y)

        if len(x[0]) ==  1:
            # グラフ表示
            plt.figure()
            plt.title('test')
            plt.plot(t, label='theacher')
            plt.plot(y, label='output')
            plt.grid()
            plt.legend()
            plt.show()

            # 平均誤差表示
            testErrer = np.average(np.abs(y - t))
            print('test errer     = {0:9.6f}'.format(testErrer))

        else:
            # 正解率表示
            correction = []
            for i in range(len(y)):
                correction.append(t[i][np.argmax(y[i])])
            accuracy = np.average(correction) * 100
            print('test accuracy     = {0:5.2f} [%]'.format(accuracy))

    def output(self, directory='param'):
        # 前回保存したファイル削除
        if os.path.isdir(directory):
            shutil.rmtree(directory)

        # ディレクトリ作成
        os.mkdir(directory)

        # パラメータ保存
        for i in range(0, len(self.layers) - 1, 2):
            self.layers[i].output(directory, i)

    class Layer:
        def __init__(self, inputNum, outputNum, batchNorm, activation):
            self.affine = self.Affine(inputNum, outputNum, activation)
            self.batchNormFlag = batchNorm
            if self.batchNormFlag == True:
                self.batchNorm = self.BatchNorm()
            if activation == 'relu':
                self.activation = self.Relu()
            elif activation == 'tanh':
                self.activation = self.Tanh()
            else:
                self.activation = self.Sigmoid()

        def forward(self, x):
            y = self.affine.forward(x)
            if self.batchNormFlag == True:
                y = self.batchNorm.forward(y)
            y = self.activation.forward(y)
            return y

        def backward(self, dy):
            dy = self.activation.backward(dy)
            if self.batchNormFlag == True:
                dy = self.batchNorm.backward(dy)
            dx = self.affine.backward(dy)
            return dx

        def update(self, learningRate, optimizer, momentum, beta1, beta2):
            self.affine.update(learningRate, optimizer,
                               momentum, beta1, beta2)
            if self.batchNormFlag == True:
                self.batchNorm.update(learningRate)

        def output(self, directory='', i=0):
            self.affine.output(directory, i)
            if self.batchNormFlag == True:
                self.batchNorm.output(directory, i)

        class Affine:
            def __init__(self, inputNum, outputNum, activation):
                # 重み
                if activation == 'relu':
                    n = np.sqrt(2/outputNum)  # Heの初期値
                else:
                    n = np.sqrt(1/outputNum)  # Xavierの初期値
                self.w = n * np.random.randn(inputNum, outputNum).astype('f8')
                self.b = n * np.random.randn(1, outputNum).astype('f8')

                self.x = None       # 入力
                self.dw = None      # 重みの勾配
                self.db = None      # バイアスの勾配

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

            def update(self, learningRate, optimizer, momentum, beta1, beta2):
                if optimizer == 'momentum':
                    self.vw = momentum * self.vw - learningRate * self.dw
                    self.vb = momentum * self.vb - learningRate * self.db
                    self.w += self.vw
                    self.b += self.vb

                elif optimizer == 'AdaGrad':
                    self.hw += self.dw**2
                    self.hb += self.db**2
                    self.w -= learningRate / self.hw ** 0.5 * self.dw
                    self.b -= learningRate / self.hb ** 0.5 * self.db

                elif optimizer == 'Adam':
                    self.mw += (1 - beta1) * (self.dw - self.mw)
                    self.mb += (1 - beta1) * (self.db - self.mb)
                    self.vw += (1 - beta2) * (self.dw ** 2 - self.vw)
                    self.vb += (1 - beta2) * (self.db ** 2 - self.vb)
                    self.w -= learningRate * self.mw / \
                        (np.sqrt(self.vw) + 1e-7)
                    self.b -= learningRate * self.mb / \
                        (np.sqrt(self.vb) + 1e-7)

                else:   # SGD
                    self.w -= learningRate * self.dw
                    self.b -= learningRate * self.db

            def output(self, directory='', i=0):
                np.savetxt(directory + '/w' + str(int(i/2 + 1)) + '.csv',
                           self.w, delimiter=',')
                np.savetxt(directory + '/b' + str(int(i/2 + 1)) + '.csv',
                           self.b, delimiter=',')

        class BatchNorm:
            def __init__(self):
                self.gamma = 1
                self.beta = 0
                self.dgamma = None
                self.dbeta = None

                self.xmu = None   # 平均０にシフトした入力データ
                self.xhat = None  # 正規化後の入力データ
                self.var = None   # 分散
                self.std = None   # 標準偏差

            def forward(self, x):
                avg = np.sum(x, axis=0) / len(x)
                self.xmu = x - avg
                self.var = np.sum(self.xmu ** 2, axis=0) / len(x)
                self.std = np.sqrt(self.var + 1e-7)
                self.xhat = self.xmu / self.std
                return self.gamma * self.xhat + self.beta

            def backward(self, dy):
                self.dbeta = np.sum(dy, axis=0)
                self.dgamma = np.sum(dy * self.xhat, axis=0)
                dxhat = self.gamma * dy
                divar = np.sum(dxhat * self.xmu, axis=0)
                dxmu1 = dxhat / self.std
                dstd = -1 / (self.std ** 2) * divar
                dvar = 0.5 * 1 / self.std * dstd
                dsq = np.ones_like(dy) / len(dy) * dvar
                dxmu2 = 2 * self.xmu * dsq
                dx1 = dxmu1 + dxmu2
                davg = -np.sum(dxmu1 + dxmu2, axis=0)
                dx2 = np.ones_like(dy) / len(dy) * davg
                dx = dx1 + dx2
                return dy

            def update(self, learningRate):
                self.gamma -= learningRate * self.dgamma
                self.beta -= learningRate * self.dbeta

            def output(self, directory, i):
                np.savetxt(directory + '/gamma' + str(int(i/2 + 1)) + '.csv',
                           self.gamma, delimiter=',')
                np.savetxt(directory + '/beta' + str(int(i/2 + 1)) + '.csv',
                           self.beta, delimiter=',')

        ########## 活性化関数 ##########

        class Sigmoid:
            def __init__(self):
                self.y = None

            def forward(self, x):
                self.y = 1 / (1 + np.exp(-x))
                return self.y

            def backward(self, dy):
                dx = self.y * (1.0 - self.y)
                return dx * dy

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

        class Tanh:
            def __init__(self):
                self.y = None

            def forward(self, x):
                self.y = np.tanh(x)
                return self.y

            def backward(self, dy):
                dx = 1 - self.y ** 2
                return dx * dy

    ########## 損失関数 ##########

    class MSE:
        def __init__(self):
            self.x = None
            self.t = None

        def forward(self, x, t):
            self.x = x
            self.t = t
            return 0.5 * np.average((x - self.t) ** 2)

        def backward(self):
            return self.x - self.t


if __name__ == '__main__':
    # 学習データ
    x = np.arange(0, 12 * np.pi, 0.001).astype('f8')
    x = x[:, np.newaxis]
    t = 0.5 * np.sin(x) + 0.5

    # ニューラルネットワーク構築
    shape = [1, 7, 6, 5, 4, 3, 1]
    NN = NewralNetwork(shape, batchNorm=False, activation='tanh')

    # 学習
    NN.learn(x, t, epoch=2000, learningRate=0.01, batchSize=1000,
             graph=True, optimizer='Adam')

    NN.test(x, t)

    # 重みファイル出力
    NN.output(directory='param')
