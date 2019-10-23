import datetime
from enum import IntEnum, auto

import matplotlib.pyplot as plt
import numpy as np

class acc(IntEnum):
    time = 0
    acc_x = auto()
    acc_y = auto()
    acc_z = auto()

# データ読み込み関数
def loaddata(filename):
    data = np.loadtxt(filename, delimiter=',',
                      skiprows=1, unpack=False, dtype=str)
    speed = []
    xyzwave = []
    for row in data:
        speed.append(row[1].astype('f8'))
        xyzwave.append(row[2: 2 + 3 * 75].astype('f8'))
    speed = np.array(speed)
    x = np.array(xyzwave)
    t = speed[:, np.newaxis]
    return x, t

filename = '0925'
accdata = np.loadtxt('data/acc/' + filename + 'acc.csv', delimiter=',',
                     skiprows=1, unpack=True, dtype=str)
acctime = []
for i in range(len(accdata[acc.time])):
    acctime.append(datetime.datetime.strptime(
        accdata[acc.time][i], '%Y/%m/%d %H:%M:%S.%f'))
acc_x = list(accdata[acc.acc_x].astype('f8'))
acc_y = list(accdata[acc.acc_y].astype('f8'))
acc_z = list(accdata[acc.acc_z].astype('f8'))

speed_x = [acc_x[0]]
speed_y = [acc_y[0]]
speed_z = [acc_z[0]]
for i in range(1, len(acctime)):
    speed_x.append(speed_x[i - 1] + acc_x[i])
    speed_y.append(speed_y[i - 1] + acc_y[i])
    speed_z.append(speed_z[i - 1] + acc_z[i])

plt.figure()
plt.plot(speed_x, label='x')
plt.plot(speed_y, label='y')
# plt.plot(speed_z, label='z')
plt.legend()
plt.show()
