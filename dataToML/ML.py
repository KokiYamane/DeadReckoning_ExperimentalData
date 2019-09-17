import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import time

# numpy行列の表示設定
np.set_printoptions(linewidth=200)

# 開始時刻
start = time.time()

# データの読み込み
filename = '0912_1815'
data = np.loadtxt(
        'MLdata/' + filename + '.txt',
        delimiter=',', skiprows=1, unpack=False, dtype=str)
speed = []
xwave = []
ywave = []
zwave = []

wavedatanum = 50
for row in data:
    speed.append(row[1].astype('float64'))
    xwave.append(row[2:2+wavedatanum].astype('float64'))
    ywave.append(row[2+wavedatanum:2+wavedatanum*2].astype('float64'))
    zwave.append(row[2+wavedatanum*2:2+wavedatanum*3].astype('float64'))

teach = speed[:, np.newaxis]
speed = np.array(speed)
xwave = np.array(xwave)
ywave = np.array(ywave)
zwave = np.array(zwave)


# 学習用関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def delta_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def loss(x, t):
    return 1/2 * np.average((x - t)**2) # MSE

# ニューラルネットワークの構造パラメータ
IMPUTNUM = wavedatanum
HIDE1NUM = 10
HIDE2NUM = 20
HIDE3NUM = 10
OUTPUTNUM = 1

# 学習パラメータ
EPOCH = 1000
LAMBDA = 0.1 # 学習率
DECAYLATE = 0.001

# 重み
w1 = np.random.rand(IMPUTNUM,HIDE1NUM)
b1 = np.random.rand(HIDE1NUM)
w2 = np.random.rand(HIDE1NUM,HIDE2NUM)
b2 = np.random.rand(HIDE2NUM)
w3 = np.random.rand(HIDE2NUM,HIDE3NUM)
b3 = np.random.rand(HIDE3NUM)
w4 = np.random.rand(HIDE3NUM,OUTPUTNUM)
b4 = np.random.rand(OUTPUTNUM)

# グラフ設定
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)
ax1.set_title('LOSS (MSE)')
line1, = ax1.plot(0)
ax1.set_ylim(0, 2)
ax1.grid()

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('output [m/s^2]')
ax2.plot(teach, label='teach')
line2, = ax2.plot(0, label='output')
line2.set_xdata(range(len(teach)))
ax2.grid()
ax2.legend()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

# 学習
inputData = xwave
losslist = []
for i in range(EPOCH):
    # 順伝播
    hide1 = np.dot(inputData, w1) + b1
    hide1s = sigmoid(hide1)
    hide2 = np.dot(hide1s, w2) + b2
    hide2s = sigmoid(hide2)
    hide3 = np.dot(hide2s, w3) + b3
    hide3s = sigmoid(hide3)
    output = np.dot(hide3s, w4) + b4

    # 損失関数計算
    losslist.append(loss(output, teach) + 1/2 * DECAYLATE * np.linalg.norm(b1, ord=2))
    if np.isnan(losslist[i]): break

    # 逆伝播
    delta_output = (output - teach)
    delta_w3 = np.dot(hide2s.T, delta_output * delta_sigmoid(hide3))
    delta_hide3 = np.dot()
    delta_w2 = delta_w3 * delta_sigmoid(hide2)
    delta_hide2 = 0
    delta_w1 = delta_w2 * delta_sigmoid(hide1)
    delta_hide1 = 0

    # 重み更新
    w4 -= LAMBDA * np.dot(hide3s.T, delta_output) / len(teach)
    b4 -= LAMBDA * np.sum(delta_output, axis=0) / len(teach)
    w3 -= LAMBDA * np.dot(hide2s.T, delta_w3) / len(teach)
    b3 -= LAMBDA * np.sum(delta_w3, axis=0) / len(teach)
    w2 -= LAMBDA * np.dot(hide1s.T, delta_w2) / len(teach)
    b2 -= LAMBDA * np.sum(delta_w2, axis=0) / len(teach)
    w1 -= LAMBDA * np.dot(inputData.T, delta_w1) / len(teach)
    b1 -= LAMBDA * np.sum(delta_w1, axis=0) / len(teach) + DECAYLATE * b1

    # 結果表示
    print('epoch = {0}  loss = {1}'.format(i, losslist[i]))

    # グラフ表示
    if i % 100 == 0:
        line1.set_xdata(range(len(losslist)))
        line1.set_ydata(losslist)
        ax1.set_xlim(0, i+1)
        line2.set_ydata(output)
        plt.draw()
        plt.pause(1e-10)


# 重みファイル出力
np.savetxt('param/w1.csv', w1, delimiter=',')
np.savetxt('param/b1.csv', b1, delimiter=',')
np.savetxt('param/w2.csv', w2, delimiter=',')
np.savetxt('param/b2.csv', b2, delimiter=',')
np.savetxt('param/w3.csv', w3, delimiter=',')
np.savetxt('param/b3.csv', b3, delimiter=',')
np.savetxt('param/w4.csv', w3, delimiter=',')
np.savetxt('param/b4.csv', b3, delimiter=',')

# 処理時間表示
elapsedTime = time.time() - start
print('elapsed time:{0:.2f}'.format(elapsedTime) + '[sec]')

# グラフ表示
plt.show()
