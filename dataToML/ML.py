import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import time

start = time.time()

# numpy行列の表示設定
np.set_printoptions(linewidth=200)

# データの読み込み
filename = "0717"
data = np.loadtxt(
        "MLdata/" + filename + ".txt",
        delimiter=",", skiprows=1, unpack=False, dtype=str)
speed = []
xwave = []
ywave = []
zwave = []
for row in data:
    speed.append(row[1].astype("float64"))
    xwave.append(row[2:36].astype("float64"))
    ywave.append(row[36:70].astype("float64"))
    zwave.append(row[70:104].astype("float64"))

speed = np.array(speed)
teach = speed[:, np.newaxis]
xwave = np.array(xwave)
ywave = np.array(ywave)
zwave = np.array(zwave)


# 学習用関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def delta_sigmoid(x):
    return x * (1-x)

def loss(x, t):
    return 1/2 * np.average((x - t)**2) # MSE

# ネットワークの構造パラメータ
IMPUTNUM = 34
HIDE1NUM = 10
HIDE2NUM = 10
OUTPUTNUM = 1

# 学習パラメータ
EPOCH = 10000
# LAMBDA = 1e-2 / len(teach) # 学習率
LAMBDA = 0.001 / len(teach) # 学習率
DECAYLATE = 0.001

# 重み
w1 = np.random.rand(IMPUTNUM,HIDE1NUM)
b1 = np.random.rand(HIDE1NUM)
w2 = np.random.rand(HIDE1NUM,HIDE2NUM)
b2 = np.random.rand(HIDE2NUM)
w3 = np.random.rand(HIDE2NUM,OUTPUTNUM)
b3 = np.random.rand(OUTPUTNUM)

# 学習
losslist = []

fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)
ax1.set_title('LOSS (MSE)')
line1, = ax1.plot(0,0)
ax1.set_ylim(0, 1)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('output')
ax2.plot(teach)
line2, = ax2.plot(0)
line2.set_xdata(range(len(teach)))
ax2.set_ylim(0, 10)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

for i in range(EPOCH):
    # 順伝播
    hide1 = np.dot(xwave, w1) + b1
    hide2 = np.dot(hide1, w2) + b2
    output = np.dot(hide2, w3) + b3

    # 損失関計算算
    losslist.append(loss(output, teach) + 1/2 * DECAYLATE * np.linalg.norm(b1, ord=2))
    if np.isnan(losslist[i]): break

    # 逆伝播
    delta_output = (output - teach)
    delta_hide2 = delta_output
    delta_hide1 = delta_hide2

    # 重み更新
    w3 -= LAMBDA * np.dot(hide2.T, delta_output)
    b3 -= LAMBDA * np.sum(delta_output, axis=0)
    w2 -= LAMBDA * np.dot(hide1.T, delta_hide2)
    b2 -= LAMBDA * np.sum(delta_hide2, axis=0)
    w1 -= LAMBDA * np.dot(xwave.T, delta_hide1)
    b1 -= LAMBDA * np.sum(delta_hide1, axis=0) + DECAYLATE * b1

    # 結果表示
    print('epoch = {0}  loss = {1}'.format(i, losslist[i]))

    if i % 100 == 0:
        line1.set_xdata(range(len(losslist)))
        line1.set_ydata(losslist)
        ax1.set_xlim(0, i+1)
        # ax1.set_ylim(0, max(losslist)*1.1)

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

elapsedTime = time.time() - start
print("elapsed time:{0:.2f}".format(elapsedTime) + "[sec]")

plt.show()
