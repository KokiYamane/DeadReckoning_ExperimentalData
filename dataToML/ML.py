import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

# numpy行列の表示設定
np.set_printoptions(linewidth=200)

# データの読み込み
filename = "0717"
data = np.loadtxt("MLdata/" + filename + ".txt", delimiter=",", skiprows=1, unpack=False, dtype=str)
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

def loss(z, t):
    return -np.sum(t * np.log(z))

# ネットワークの構造パラメータ
IMPUTNUM = 34
HIDE1NUM = 10
HIDE2NUM = 10
OUTPUTNUM = 1

# 学習パラメータ
EPOCH = 100000
LAMBDA = 1e-6 / len(teach) # 学習率

# 重み
w1 = np.random.rand(IMPUTNUM,HIDE1NUM)
b1 = np.random.rand(HIDE1NUM)
w2 = np.random.rand(HIDE1NUM,HIDE2NUM)
b2 = np.random.rand(HIDE2NUM)
w3 = np.random.rand(HIDE2NUM,OUTPUTNUM)
b3 = np.random.rand(OUTPUTNUM)

# 学習
losslist = []
plt.figure()
line, = plt.plot(0,0)
for i in range(EPOCH):
    # 順伝播
    hide1 = np.dot(xwave, w1) + b1
    hide1s = sigmoid(hide1)
    hide2 = np.dot(hide1s, w2) + b2
    hide2s = sigmoid(hide2)
    output = np.dot(hide2s, w3) + b3
    outputs = sigmoid(output)

    # 損失関計算算
    losslist.append(loss(outputs, teach))
    if np.isnan(losslist[i]): break

    # 逆伝播
    delta_output = (outputs - teach) * delta_sigmoid(output)
    delta_hide2 = delta_output * delta_sigmoid(hide2)
    delta_hide1 = delta_hide2 * delta_sigmoid(hide1)

    # 重み更新
    w3 -= LAMBDA * np.dot(hide2s.T, delta_output)
    b3 -= LAMBDA * np.sum(delta_output, axis=0)
    w2 -= LAMBDA * np.dot(hide1s.T, delta_hide2)
    b2 -= LAMBDA * np.sum(delta_hide2, axis=0)
    w1 -= LAMBDA * np.dot(xwave.T, delta_hide1)
    b1 -= LAMBDA * np.sum(delta_hide1, axis=0)

    # 結果表示
    print('epoch = ' + str(i) + '  loss = ' + str(losslist[i]))
    # print(b1)
    if i % 100 == 0:
        line.set_xdata(range(len(losslist)))
        line.set_ydata(losslist)
        plt.xlim(0,i+1)
        plt.ylim(0,max(losslist)*1.1)
        plt.draw()
        plt.pause(1e-10)


# 重みファイル出力
np.savetxt('param/w1.csv', w1, delimiter=',')
np.savetxt('param/b1.csv', b1, delimiter=',')
np.savetxt('param/w2.csv', w2, delimiter=',')
np.savetxt('param/b2.csv', b2, delimiter=',')
np.savetxt('param/w3.csv', w3, delimiter=',')
np.savetxt('param/b3.csv', b3, delimiter=',')

plt.show()