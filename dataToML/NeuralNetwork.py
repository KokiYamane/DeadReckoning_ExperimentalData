import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import shutil

class NewralNetwork:
    def __init__(self, inputdata, theacherdata, shape=[1,3,1], epoch=1000, learninglate=0.1):
        self.inputdata = inputdata          # 入力データ
        self.teacherdata = theacherdata     # 教師データ
        self.losslist = []                  # 損失関数
        self.outputdata = None              # 出力

        # ハイパーパラメータ
        self.epoch = epoch
        self.learninglate = learninglate

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

    def learn(self):
        for i in range(self.epoch):
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
                dy = self.layers[j].update(self.learninglate)

            # グラフ表示
            if (i+1) % 100 == 0: self.show()

        # グラフ表示
        plt.show()

    def show(self):
        # グラフ１
        self.ax1.cla()
        self.ax1.set_title('LOSS')
        self.ax1.set_xlim(-self.epoch * 0.05, self.epoch * 1.05)
        self.ax1.set_ylim(-max(self.losslist) * 0.05, max(self.losslist) * 1.05)
        self.ax1.grid()
        self.ax1.plot(self.losslist)

        # グラフ２
        self.ax2.cla()
        self.ax2.set_title('output')
        self.ax2.grid()
        self.ax2.plot(range(len(self.inputdata)), self.teacherdata, label='teacher')
        self.ax2.plot(range(len(self.inputdata)), self.outputdata, label='output')
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
            self.w = np.random.rand(imputnum, outputnum)
            self.b = np.random.rand(outputnum)
            self.x = None
            self.dw = None
            self.db = None

        def forward(self, x):
            self.x = x
            return np.dot(x, self.w) + self.b

        def backward(self, dy):
            self.dw = np.dot(self.x.T, dy)
            self.db = np.sum(dy, axis=0)
            return np.dot(dy, self.w.T)

        def update(self, LEARNINGLATE):
            self.w -= LEARNINGLATE * self.dw
            self.b -= LEARNINGLATE * self.db

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
            dy[self.mask] = 0
            dx = dy
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
    NN = NewralNetwork(x, t, shape, 10000, 0.1)

    # 学習
    NN.learn()

    # 重みファイル出力
    NN.output()
