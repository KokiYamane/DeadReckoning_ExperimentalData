import datetime
from enum import IntEnum, auto

import matplotlib.pyplot as plt
import numpy as np

# データの項目
class acc(IntEnum):
    time = 0
    acc_x = auto()
    acc_y = auto()
    acc_z = auto()
    gyro_x = auto()
    gyro_y = auto()
    gyro_z = auto()
    angle_x = auto()
    angle_y = auto()
    angle_z = auto()
    step = auto()


class rtk(IntEnum):
    date = 0
    time = auto()
    latitude = auto()
    longitude = auto()


def deg2rad(x):
    return x * np.pi/180


def calcDistance(latitude1, longitude1, latitude2, longitude2):
    GRS80_A = 6378137.000  # 長半径 a(m)
    GRS80_E2 = 0.00669438002301188  # 第一遠心率  eの2乗
    R = 6378137  # 赤道半径[m]

    my = deg2rad((latitude1 + latitude2) / 2.0)

    # 卯酉線曲率半径を求める(東と西を結ぶ線の半径)
    sinMy = np.sin(my)
    w = np.sqrt(1.0 - GRS80_E2 * sinMy * sinMy)
    n = GRS80_A / w

    # 子午線曲線半径を求める(北と南を結ぶ線の半径)
    mnum = GRS80_A * (1 - GRS80_E2)
    m = mnum / (w * w * w)

    deltaLatitude = deg2rad(latitude2 - latitude1)
    deltaLongitude = deg2rad(longitude2 - longitude1)

    deltaX = n * np.cos(my) * deltaLongitude
    deltaY = m * deltaLatitude
    return (deltaX**2 + deltaY**2)**0.5


def timeSub(time1, time2):
    return (time1 - time2).total_seconds()


def loadAccData(filename):
    accdata = np.loadtxt(filename, delimiter=',', skiprows=1,
                         unpack=True, dtype=str)
    acctime = []
    for i in range(len(accdata[acc.time])):
        acctime.append(datetime.datetime.strptime(
            accdata[acc.time][i], '%Y/%m/%d %H:%M:%S.%f'))
    accdict = {}
    accdict['time'] = list(acctime)
    accdict['acc_x'] = list(accdata[acc.acc_x].astype('f8'))
    accdict['acc_y'] = list(accdata[acc.acc_y].astype('f8'))
    accdict['acc_z'] = list(accdata[acc.acc_z].astype('f8'))
    accdict['gyro_x'] = list(accdata[acc.gyro_x].astype('f8'))
    accdict['gyro_y'] = list(accdata[acc.gyro_y].astype('f8'))
    accdict['gyro_z'] = list(accdata[acc.gyro_z].astype('f8'))
    accdict['angle_x'] = list(accdata[acc.angle_x].astype('f8'))
    accdict['angle_y'] = list(accdata[acc.angle_y].astype('f8'))
    accdict['angle_z'] = list(accdata[acc.angle_z].astype('f8'))
    accdict['step'] = list(accdata[acc.step].astype('f8'))
    accdict['stepflag'] = [0]
    for i in range(1, len(accdict['step'])):
        accdict['stepflag'].append(accdict['step'][i] - accdict['step'][i-1])
    return accdict


def loadRTKData(filename):
    rtkdata = np.loadtxt(filename, delimiter=',', skiprows=1,
                         unpack=True, dtype=str)
    latitude = rtkdata[rtk.latitude].astype('f8')
    longitude = rtkdata[rtk.longitude].astype('f8')
    rtkdict = {}
    rtkdict['time'] = []
    rtkdict['speed'] = []
    for i in range(len(rtkdata[rtk.date])):
        # 時刻
        time = rtkdata[rtk.date][i] + ' ' + rtkdata[rtk.time][i]
        time = datetime.datetime.strptime(time, '%Y/%m/%d %H:%M:%S.%f')
        time += datetime.timedelta(hours=9)
        time -= datetime.timedelta(seconds=18)
        if i == 0:
            rtkdict['starttime'] = time
            continue
        rtkdict['time'].append(time)

        # 速度算出
        if i == 1:
            elapsedTime = timeSub(rtkdict['time'][i-1], rtkdict['starttime'])
        else:
            elapsedTime = timeSub(rtkdict['time'][i-1], rtkdict['time'][i-2])
        speed = calcDistance(latitude[i-1], longitude[i-1],
                             latitude[i],   longitude[i]) / elapsedTime
        rtkdict['speed'].append(speed)
    return rtkdict


# データの読み込み
# filename = '0912_1800'
# filename = '0912_1815'
# filename = '0925'
filename = '1010'

accdata = loadAccData('acc/' + filename + 'acc.csv')
rtkdata = loadRTKData('rtk/csv/' + filename + 'rtk.csv')

# 外れ値削除
threshold = 3
i = len(rtkdata['speed'])-1
while i >= 0:
    if rtkdata['speed'][i] > threshold:
        del rtkdata['time'][i]
        del rtkdata['speed'][i]
    i -= 1

# 線形補完
i = len(rtkdata['speed'])-1
while i >= 0:
    timediff = timeSub(rtkdata['time'][i], rtkdata['time'][i-1])
    if timediff > 0.2:
        nowtime = rtkdata['time'][i] - datetime.timedelta(seconds=1)
        rtkdata['time'].insert(i, nowtime)
        speeddiff = rtkdata['speed'][i] - rtkdata['speed'][i-1]
        nowacc = speeddiff / timediff
        nowspeed = rtkdata['speed'][i] - nowacc * (timediff-0.2)
        rtkdata['speed'].insert(i, nowspeed)
    else:
        i -= 1

# 5[Hz] から 1[Hz] に
speedlist = []
frequency = 5  # [Hz]
for i in range(int(len(rtkdata['speed']) / frequency)):
    if i < frequency:
        speedlist.append(0.0)
        continue
    partialSum = 0
    for j in range(frequency):
        partialSum += rtkdata['speed'][frequency*i-j]
    average = partialSum / frequency
    speedlist.append(average)

# 加速度データの不要な部分削除
i = len(accdata['time'])-1
while i >= 0:
    if accdata['time'][i] < rtkdata['starttime']:
        for col in accdata.values():
            del col[i]
    i -= 1

print(accdata['time'][0])
print(rtkdata['starttime'])
print(rtkdata['time'][0])

# ファイル書き込み
datanum = 50
f = open('ML/' + filename + '.csv', mode='w')
f.write('time[s], speed[m/s], accwave_x({0})[G], accwave_y({0})[G],'
        'accwave_z({0})[G]\n'.format(datanum))
for i in range(len(speedlist)):
    if datanum * (i + 1) > len(accdata['time']):
        break
    f.write(str(i) + ', ' + str(speedlist[i]) + ', ')
    for j in range(datanum):
        f.write(str(accdata['acc_x'][datanum*i+j] / 9.8) + ', ')
    for j in range(datanum):
        f.write(str(accdata['acc_y'][datanum*i+j] / 9.8) + ', ')
    for j in range(datanum):
        f.write(str(accdata['acc_z'][datanum*i+j] / 9.8) + ', ')
    f.write('\n')
f.close

# グラフ表示
i = 1100
renge = 50
acc_x = list(map(lambda x: x / 9.8, accdata['acc_x']))
acc_y = list(map(lambda x: x / 9.8, accdata['acc_y']))
acc_z = list(map(lambda x: x / 9.8, accdata['acc_z']))
plt.plot(accdata['time'], accdata['stepflag'], marker='.')
plt.plot(accdata['time'], acc_x, marker='.')
plt.plot(accdata['time'], acc_y, marker='.')
plt.plot(accdata['time'], acc_z, marker='.')
plt.plot(rtkdata['time'], rtkdata['speed'], marker='.')
plt.grid()
plt.show()
