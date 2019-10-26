import datetime
from enum import IntEnum, auto

import matplotlib.pyplot as plt
import numpy as np

# 定数
X = 0
Y = 1
Z = 2

# データの項目
class item(IntEnum):
    time = 0
    accX = auto()
    accY = auto()
    accZ = auto()
    gyroX = auto()
    gyroY = auto()
    gyroZ = auto()
    angleX = auto()
    angleY = auto()
    angleZ = auto()
    step = auto()


# データの読み込み
filename = '0912_1800'
# filename = '0912_1815'
# filename = '0925'
# filename = '1010'

data = np.loadtxt('acc/' + filename + 'acc.csv', delimiter=',',
                  skiprows=1, unpack=True, dtype=str)
time = []
acc = []
gyro = []
angleByMag = []
for i in range(len(data[item.time])):
    time.append(datetime.datetime.strptime(
        data[item.time][i], '%Y/%m/%d %H:%M:%S.%f'))
acc.append(list(data[item.accX].astype('f8')))
acc.append(list(data[item.accY].astype('f8')))
acc.append(list(data[item.accZ].astype('f8')))
gyro.append(list(data[item.gyroX].astype('f8')))
gyro.append(list(data[item.gyroY].astype('f8')))
gyro.append(list(data[item.gyroZ].astype('f8')))
angleByMag.append(list(data[item.angleX].astype('f8')))
angleByMag.append(list(data[item.angleY].astype('f8')))
angleByMag.append(list(data[item.angleZ].astype('f8')))
step = list(data[item.step].astype('f8'))

# 時系列処理
angleByGyro = []
angleByGyro.append(list([gyro[X][0]]))
angleByGyro.append(list([gyro[Y][0]]))
angleByGyro.append(list([gyro[Z][0]]))
Xlist = [0.0]
Ylist = [0.0]
for i in range(1, len(gyro[X])):
    for j in range(3):
        angle = angleByGyro[j][i-1] + gyro[j][i]
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        angleByGyro[j].append(angle)

    if step[i] - step[i-1] > 0:
        stride = 0.5
        Xlist.append(Xlist[-1] + stride * np.cos(angleByMag[Z][i]))
        Ylist.append(Ylist[-1] + stride * np.sin(angleByMag[Z][i]))

plt.figure()
plt.plot(Xlist, Ylist)
plt.axes().set_aspect('equal')

# plt.plot(angleByGyro[X])
# # plt.plot(angleByGyro[Y])
# # plt.plot(angleByGyro[Z])
# # plt.plot(angleByMag[X])
# # plt.plot(angleByMag[Y])
# angleByMag[Z] = map(lambda x: x * -180/np.pi, angleByMag[Z])
# plt.plot(list(angleByMag[Z]))

plt.show()
