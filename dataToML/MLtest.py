import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

# numpy行列の表示設定
np.set_printoptions(linewidth=200)

# データの読み込み
filename = "0716"
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

# 重み
w1 = np.loadtxt("param/w1.csv", delimiter=",")
b1 = np.loadtxt("param/b1.csv", delimiter=",")
w2 = np.loadtxt("param/w2.csv", delimiter=",")
b2 = np.loadtxt("param/b2.csv", delimiter=",")
w3 = np.loadtxt("param/w3.csv", delimiter=",")
b3 = np.loadtxt("param/b3.csv", delimiter=",")

# 順伝播
hide1 = np.dot(xwave, w1) + b1
hide2 = np.dot(hide1, w2) + b2
output = np.dot(hide2, w3) + b3

# 損失関計算算
output = output[:, np.newaxis]
errer = teach - output
print(np.average(errer))
print(np.average(teach))

plt.figure()
plt.plot(teach)
plt.plot(output)
plt.show()