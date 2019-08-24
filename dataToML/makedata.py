import numpy as np
import matplotlib.pyplot as plt

# スマホ加速度データの項目
from enum import IntEnum, auto
class item():
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

# スマホ加速度データの読み込み
filename = "0716acc"
accdata = np.loadtxt("accdata/" + filename + ".csv", delimiter=",", skiprows=2, unpack=True) 

# RTKデータの読み込み
filename = "0716rtk"
rtkdata = np.loadtxt(
    "rtkdata/" + filename + ".csv",
    delimiter=",", skiprows=1, unpack=True,
    # dtype=[
    #     ("date", "S10"),
    #     ("time", "str"),
    #     ("latitude", "f8"),
    #     ("longitude", "f8"),
    #     ("height", "f8"),
    #     ("Q", "i8"),
    #     ("ns", "i8"),
    #     ("sdn", "f8"),
    #     ("sde", "f8"),
    #     ("sdu", "f8"),
    #     ("sdne", "f8"),
    #     ("sdeu", "f8"),
    #     ("sdun", "f8"),
    #     ("age", "f8"),
    #     ("ratio", "f8"),
    # ],
    dtype=str,
)

# 時刻
import datetime
date = rtkdata[0][0] + " " + rtkdata[1][0]
print(datetime)
datetime = datetime.datetime.strptime(date, '%Y/%m/%d %H:%M:%S.%f')
print(datetime)

# 速度
deltaLatitude  = (rtkdata[item.rtk.latitude][1].astype(np.float32) - rtkdata[item.rtk.latitude][0]).astype(np.float32) * np.pi/180
deltaLongitude = (rtkdata[item.rtk.longitude][1].astype(np.float32) - rtkdata[item.rtk.longitude][0].astype(np.float32)) * np.pi/180
R = 6378137 # 赤道半径[m]
deltaX = R * deltaLongitude * np.cos(rtkdata[item.rtk.longitude][1])
deltaY = R * deltaLatitude

print(deltaX)
print(deltaY)

# # 画面生成
# fig = plt.figure(figsize=(6, 6))

# # グラフ生成
# graph = fig.add_subplot(111)

# # プロット
# graph.plot(rtkdata[item.rtk.latitude], rtkdata[item.rtk.longitude], color="red", label="gyro")

# # 軸の比率を設定
# graph.set_aspect("equal")

# graph.tick_params(labelbottom=False, bottom=False)
# graph.tick_params(labelleft=False, bottom=False)

# # 表示
# plt.show()
