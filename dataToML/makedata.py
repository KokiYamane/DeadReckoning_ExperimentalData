import datetime
from enum import IntEnum, auto

import numpy as np

# データの項目
class acc(IntEnum):
    time = 0
    acc_x = auto()
    acc_y = auto()
    acc_z = auto()


class rtk(IntEnum):
    date = 0
    time = auto()
    latitude = auto()
    longitude = auto()
    height = auto()


# データの読み込み
filename = '0912_1815'

accdata = np.loadtxt('data/acc/' + filename + 'acc.csv', delimiter=',',
                     skiprows=1, unpack=True, dtype=str)
acctime = []
for i in range(len(accdata[acc.time])):
    acctime.append(datetime.datetime.strptime(
        accdata[acc.time][i], '%Y/%m/%d %H:%M:%S.%f'))
acc_x = list(accdata[acc.acc_x].astype('f8'))
acc_y = list(accdata[acc.acc_y].astype('f8'))
acc_z = list(accdata[acc.acc_z].astype('f8'))

rtkdata = np.loadtxt('data/rtk/csv/' + filename + 'rtk.csv', delimiter=',',
                     skiprows=1, unpack=True, dtype=str)
latitude = rtkdata[rtk.latitude].astype('f8')
longitude = rtkdata[rtk.longitude].astype('f8')

# RTKデータ処理
timelist = []
speedlist = []
for i in range(len(rtkdata[rtk.date])):
    # 時刻
    time = rtkdata[rtk.date][i] + ' ' + rtkdata[rtk.time][i]
    time = datetime.datetime.strptime(time, '%Y/%m/%d %H:%M:%S.%f')
    if i == 0:
        rtkstarttime = time
        continue
    else:
        timedelta = time - rtkstarttime
        timelist.append(timedelta.total_seconds())

    # 速度算出
    deltaLatitude = (latitude[i] - latitude[i-1]) * np.pi/180
    deltaLongitude = (longitude[i] - longitude[i-1]) * np.pi/180
    R = 6378137  # 赤道半径[m]
    deltaX = R * deltaLongitude * np.cos(longitude[i])
    deltaY = R * deltaLatitude
    if i == 1:
        elapsedTime = timelist[i-1]
    else:
        elapsedTime = timelist[i-1] - timelist[i-2]
    speed = (deltaX**2 + deltaY**2)**0.5 / elapsedTime
    speedlist.append(speed)

# 外れ値削除
i = len(speedlist)-1
while i >= 0:
    if speedlist[i] > 5:
        del timelist[i]
        del speedlist[i]
    i -= 1

# 線形補完
i = len(speedlist)-1
while i >= 0:
    timediff = timelist[i] - timelist[i-1]
    speeddiff = speedlist[i] - speedlist[i-1]
    if timediff > 1:
        timelist.insert(i, timelist[i]-1)
        nowspeed = speedlist[i] - speeddiff / timediff * (timediff-1)
        speedlist.insert(i, nowspeed)
    else:
        i -= 1

# 5[Hz] から 1[Hz] に
speedlist_LPF = []
param = 5  # [Hz]
for i in range(int(len(speedlist)/param)):
    if i < param:
        speedlist_LPF.append(0.0)
        continue
    partialSum = 0
    for j in range(param):
        partialSum += speedlist[param*i-j]
    speedlist_LPF.append(partialSum / param)

# 加速度データの不要な部分削除
timedelta = rtkstarttime - acctime[0]
timediff = timedelta.total_seconds() + 60*60*9
timediff = round(timediff, 3)
i = len(acctime)-1
while i >= 0:
    if acctime[i] < rtkstarttime:
        del acctime[i]
        del acc_x[i]
        del acc_y[i]
        del acc_z[i]
    i -= 1

# ファイル書き込み
datanum = 50
f = open('data/ML/' + filename + '.csv', mode='w')
f.write('time[s], speed[m/s], accwave_x({0})[G], accwave_y({0})[G],'
        'accwave_z({0})[G]\n'.format(datanum))
for i in range(len(speedlist_LPF)):
    if datanum * (i + 1) > len(acc_x):
        break
    f.write(str(i) + ', ' + str(speedlist_LPF[i]) + ', ')
    for j in range(datanum):
        f.write(str(acc_x[datanum*i+j] / 9.8) + ', ')
    for j in range(datanum):
        f.write(str(acc_y[datanum*i+j] / 9.8) + ', ')
    for j in range(datanum):
        f.write(str(acc_z[datanum*i+j] / 9.8) + ', ')
    f.write('\n')
f.close
