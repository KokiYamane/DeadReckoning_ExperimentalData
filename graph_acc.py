import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Yu Gothic"]

# データの項目
from enum import IntEnum, auto
class item(IntEnum):
    time = 0
    acc_x = auto()
    acc_y = auto()
    acc_z = auto()
    vel_x = auto()
    vel_y = auto()
    vel_z = auto()
    pos_x = auto()
    pos_y = auto()
    pos_z = auto()
    rad_vel_x = auto()
    rad_vel_y = auto()
    rad_vel_z = auto()
    rad_gyro_x = auto()
    rad_gyro_y = auto()
    rad_gyro_z = auto()
    rad_mag_x = auto()
    rad_mag_y = auto()
    rad_mag_z = auto()
    pos_step_gyro_x = auto()
    pos_step_gyro_y = auto()
    pos_step_mag_x = auto()
    pos_step_mag_y = auto()
    pos_step_x = auto()
    pos_step_y = auto()

# ローパスフィルタ
def LPF(list):
    list_LPF = [ list[0] ]
    i = 1
    C = 0.1
    for nowdata in list:
        list_LPF.append( (1-C) * list_LPF[i-1] + C * nowdata )
        i+=1
    list_LPF.pop(i-1)
    return list_LPF

# データの読み込み
filename = "0716acc"
data = np.loadtxt("data/" + filename + ".csv", delimiter=",", skiprows=2, unpack=True) 

# グラフ作成
fig = plt.figure(figsize=(6, 6))

graph1 = fig.add_subplot(311) 
graph2 = fig.add_subplot(312)
graph3 = fig.add_subplot(313)

graph1.plot(data[item.time], data[item.acc_x], color="black")
# graph1.plot(data[item.time], LPF(data[item.acc_x]))
graph2.plot(data[item.time], data[item.acc_y], color="black")
# graph2.plot(data[item.time], LPF(data[item.acc_y]))
graph3.plot(data[item.time], data[item.acc_z], color="black")
# graph3.plot(data[item.time], LPF(data[item.acc_z]))

# 値の範囲
xmin = 500
xmax = xmin + 1.5
ymin = -15
ymax = -ymin

graph1.set_xlim(xmin,xmax)
graph2.set_xlim(xmin,xmax)
graph3.set_xlim(xmin,xmax)

graph1.set_ylim(ymin,ymax)
graph2.set_ylim(ymin,ymax)
graph3.set_ylim(ymin,ymax)

# ラベル
graph1.tick_params(labelbottom=False, bottom=False)
graph2.tick_params(labelbottom=False, bottom=False)
graph3.set_xlabel("時間 [秒]")

graph1.set_ylabel("x [m/s^2]")
graph2.set_ylabel("y [m/s^2]")
graph3.set_ylabel("z [m/s^2]")

#グリッド
graph1.grid()
graph2.grid()
graph3.grid()

fig.savefig("graph/acc/" + filename + ".png")

# plt.show()
