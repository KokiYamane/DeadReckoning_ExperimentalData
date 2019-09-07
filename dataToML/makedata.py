import numpy as np

# データの項目
from enum import IntEnum, auto
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
filename = "0716"
accdata = np.loadtxt("accdata/" + filename + "acc.csv", delimiter=",", skiprows=2, unpack=True)
acctime = list(accdata[acc.time])
acc_x = list(accdata[acc.acc_x])
acc_y = list(accdata[acc.acc_y])
acc_z = list(accdata[acc.acc_z])
rtkdata = np.loadtxt("rtkdata/" + filename + "rtk.csv", delimiter=",", skiprows=1, unpack=True, dtype=str)
latitude = rtkdata[rtk.latitude].astype("float32")
longitude = rtkdata[rtk.longitude].astype("float32")

import datetime
timelist = []
speedlist = []
for i in range(len(rtkdata[0])):
    # 時刻
    time = rtkdata[rtk.date][i] + " " + rtkdata[rtk.time][i]
    time = datetime.datetime.strptime(time, "%Y/%m/%d %H:%M:%S.%f")
    if i == 0:
        rtkstarttime = time
        continue
    else:
        timedelta = time - rtkstarttime
        timelist.append(timedelta.total_seconds())

    # 速度
    deltaLatitude  = (latitude[i] - latitude[i-1]) * np.pi/180
    deltaLongitude = (longitude[i] - longitude[i-1]) * np.pi/180
    R = 6378137 # 赤道半径[m]
    deltaX = R * deltaLongitude * np.cos(longitude[i])
    deltaY = R * deltaLatitude
    if i == 1: elapsedTime = timelist[i-1]
    else:      elapsedTime = timelist[i-1] - timelist[i-2]
    speed = (deltaX**2 + deltaY**2)**0.5 / elapsedTime
    speedlist.append(speed)

# 外れ値削除
i = len(speedlist)-1
while i >= 0:
    if speedlist[i] > 10:
        del timelist[i]
        del speedlist[i]
    i-=1

# 線形補完
i = len(speedlist)-1
while i >= 0:
    timediff = timelist[i] - timelist[i-1]
    speeddiff = speedlist[i] - speedlist[i-1]
    if timediff > 1:
        timelist.insert(i,timelist[i]-1)
        nowspeed = speedlist[i] - speeddiff / timediff * (timediff-1)
        speedlist.insert(i, nowspeed)
    else: i-=1

f = open("accdata/" + filename + "acc.csv")
line = f.readline()
accstarttime = datetime.datetime.strptime(line, "%Y/%m/%d %H:%M:%S.%f ")
f.close

timedelta = rtkstarttime - accstarttime
timediff = timedelta.total_seconds() + 60*60*9
timediff = round(timediff, 3)
i = len(acctime)-1
while i >= 0:
    if acctime[i] < timediff:
        del acctime[i]
        del acc_x[i]
        del acc_y[i]
        del acc_z[i]
    else: 
        acctime[i] -= timediff
        acctime[i] = round(acctime[i], 3)
    i-=1

# ファイル書き込み
f = open("MLdata/" + filename + ".txt", mode="w")
f.write("time[s], speed[m/s], accwave_x(34)[G], accwave_y(34)[G], accwave_z(34)[G\n")
for i in range(len(timelist)):
    if 34* (i + 1) > len(acc_x): break
    f.write(str(timelist[i]) + ", " + str(speedlist[i]) + ", ")
    for j in range(34):
        f.write(str(acc_x[34*i+j] / 9.8) + ", ")
    for j in range(34):
        f.write(str(acc_y[34*i+j] / 9.8) + ", ")
    for j in range(34):
        f.write(str(acc_z[34*i+j] / 9.8) + ", ")
    f.write("\n")
f.close
