import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import shutil
import time

class NewralNetwork:
    def __init__(self, inputdata, theacherdata, shape=[1,3,1]):
        self.inputdata = inputdata          # 入力データ
        self.teacherdata = theacherdata     # 教師データ
        self.losslist = []                  # 損失関数
        self.outputdata = None              # 出力

        # ニューラルネットワーク
        self.layers = []
        for i in range(len(shape)-1):
            self.layers.append(self.Affine(shape[i], shape[i+1]))
            self.layers.append(self.Sigmoid())
        self.layers.append(self.MSE(self.teacherdata))

        # グラフ
        self.fig = plt.figure(figsize=(10,5))
        self.ax1 = self.fig.add_subplot(1,2,1)
        self.ax2 = self.fig.add_subplot(1,2,2)

    def learn(self, epoch=1000, method='SGD', learningRate=0.1, momentum=0.9, beta1=0.9, beta2=0.999):
        start = time.time()
        for i in range(epoch):
            # 準伝播
            y = self.inputdata
            for j in range(len(self.layers)):
                y = self.layers[j].forward(y)
                if j == len(self.layers)-2: self.outputdata = y
                if j == len(self.layers)-1: self.losslist.append(y)

            # 逆伝播
            dy = None
            for j in reversed(range(len(self.layers))):
                if j == len(self.layers)-1: dy = self.layers[j].backward()
                else: dy = self.layers[j].backward(dy)

            # 重み更新
            for j in range(0, len(self.layers)-1, 2):
                dy = self.layers[j].update(method, learningRate, momentum, beta1, beta2)

            # グラフ表示
            if (i+1) % 100 == 0: self.show(epoch)

        # 経過時間表示
        elapsedTime = time.time() - start
        print('learning time  = {0:9.2f} [sec]'.format(elapsedTime))

        # 平均誤表示示
        learningErrer = np.average(self.outputdata - self.teacherdata) * 5
        print('learning errer = {0:9.6f} [m/s]'.format(learningErrer))

        # グラフ表示
        plt.show()

    def test(self, inputdata, teacherdata):
        y = inputdata
        for j in range(len(self.layers)-1):
            y = self.layers[j].forward(y)
            if j == len(self.layers)-2: outputdata = y

        plt.figure()
        plt.title('test')
        plt.plot(teacherdata, label='theacher')
        plt.plot(outputdata, label='output')
        plt.legend()
        plt.show()

        # 平均誤表示示
        testErrer = np.average(outputdata - teacherdata) * 5
        print('test errer     = {0:9.6f} [m/s]'.format(testErrer))

        return outputdata

    def show(self, epoch):
        # グラフ１
        self.ax1.cla()
        self.ax1.set_title('LOSS')
        self.ax1.set_xlim(-epoch * 0.05, epoch * 1.05)
        self.ax1.set_ylim(-max(self.losslist) * 0.05, max(self.losslist) * 1.05)
        self.ax1.grid()
        self.ax1.plot(self.losslist)

        # グラフ２
        self.ax2.cla()
        self.ax2.set_title('output')
        self.ax2.grid()
        self.ax2.plot(self.teacherdata, label='teacher')
        self.ax2.plot(self.outputdata, label='output')
        self.ax2.legend()

        plt.draw()
        plt.pause(1e-10)

    def output(self, directory='param'):
        if os.path.isdir(directory): shutil.rmtree(directory)
        os.mkdir(directory)
        for i in range(0, len(self.layers)-1, 2):
            self.layers[i].output(directory, i)

    class Affine:
        def __init__(self, imputnum, outputnum):
            # 重み
            self.w = 0.01 * np.random.rand(imputnum, outputnum)
            self.b = 0.01 * np.random.rand(1, outputnum)

            # 入力
            self.x = None

            # 勾配
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

        def update(self, method='SGD', learningRate=0.1, momentum=0.9, beta1=0.9, beta2=0.999):
            if method == 'momentum':
                self.vw = momentum * self.vw - learningRate * self.dw
                self.vb = momentum * self.vb - learningRate * self.db
                self.w += self.vw
                self.b += self.vb

            elif method == 'AdaGrad':
                self.hw += self.dw**2
                self.hb += self.db**2
                self.w -= learningRate / self.hw**0.5 * self.dw
                self.b -= learningRate / self.hb**0.5 * self.db

            elif method == 'Adam':
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
            np.savetxt(directory + '/w' + str(int(i/2+1)) + '.csv', self.w, delimiter=',')
            np.savetxt(directory + '/b' + str(int(i/2+1)) + '.csv', self.b, delimiter=',')

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

        def forward(self, x):
            self.mask = (x <= 0)
            y = x.copy()
            y[self.mask] = 0
            return y

        def backward(self, dy):
            dx = dy.copy()
            dx[self.mask] = 0
            return dx

    class MSE:
        def __init__(self, t):
            self.t = t
            self.x = None

        def forward(self, x):
            self.x = x
            return 0.5 * np.average((x - self.t)**2)

        def backward(self):
            return self.x - self.t

if __name__ == '__main__':
    # 学習データ
    x = np.arange(0, 6*np.pi, 0.1)
    x = x[:, np.newaxis]
    t = 0.5 * np.sin(x) + 0.5

    # ニューラルネットワーク構築
    shape = [1, 3, 1]
    NN = NewralNetwork(x, t, shape)

    # 学習
    NN.learn(epoch=10000, method='Adam', learningRate=0.01)

    # 重みファイル出力
    NN.output()
