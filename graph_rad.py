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
filename = "turnTable2"
data = np.loadtxt("data/" + filename + ".txt", delimiter = ",", skiprows = 2, unpack = True) 

# 角度理想値
expected = (data[item.time]+25)*6.12%360-180

# 誤差算出
error_gyro = (expected - data[item.rad_gyro_z]*180/np.pi) % 360
i=0
for deg in error_gyro:
    if deg > 180:
        error_gyro[i] = deg-360
    i+=1
error_mag = (expected - data[item.rad_mag_z]*180/np.pi) % 360
i=0
for deg in error_mag:
    if deg > 180:
        error_mag[i] = deg-360
    i+=1

# 画面生成
fig = plt.figure(figsize=(6, 6))

# グラフ生成
graph1 = fig.add_subplot(311) 
graph2 = fig.add_subplot(312)
graph3 = fig.add_subplot(313)

# プロット
graph1.plot(data[item.time]/60, data[item.rad_vel_z]*180/np.pi, color="gray", label = "raw data")
graph1.plot(data[item.time]/60, LPF(data[item.rad_vel_z]*180/np.pi), color="black", label = "LPF")

graph2.plot(data[item.time]/60, expected, color="gray", label = "expected")
graph2.plot(data[item.time]/60, data[item.rad_gyro_z]*180/np.pi, color="red", label = "gyro")
graph2.plot(data[item.time]/60, data[item.rad_mag_z]*180/np.pi, color="blue", label = "mag")

graph3.plot(data[item.time]/60, error_gyro, color="red", label = "gyro")
graph3.plot(data[item.time]/60, error_mag, color="blue", label = "mag")

# 値の範囲
xmin = 0
xmax = 5
graph1.set_xlim(xmin,xmax)
graph2.set_xlim(xmin,xmax)
graph3.set_xlim(xmin,xmax)

ymin = -180
ymax =  180
graph2.set_ylim(ymin,ymax)
graph3.set_ylim(ymin,ymax)

# 目盛
graph2.set_yticks(np.linspace(-180, 180, 9))
graph3.set_yticks(np.linspace(-180, 180, 9))

# ラベル
graph1.tick_params(labelbottom=False, bottom=False)
graph2.tick_params(labelbottom=False, bottom=False)
graph3.set_xlabel("時間 [分]")

graph1.set_ylabel("角速度 [deg/s]")
graph2.set_ylabel("角度 [deg]")
graph3.set_ylabel("誤差 [deg]")

graph1.yaxis.set_label_coords(-0.1, 0.5)
graph2.yaxis.set_label_coords(-0.1, 0.5)
graph3.yaxis.set_label_coords(-0.1, 0.5)

# グリッド
graph1.grid()
graph2.grid()
graph3.grid()

# 凡例
# graph1.legend(bbox_to_anchor=(0, 0), loc="lower left", borderaxespad=0, fontsize=12)

# 注釈
graph1.annotate("生データ",
                xy=[4, 5],
                xytext=[4, 1.5],
                arrowprops=dict(arrowstyle="->"))
graph1.annotate("LPF後のデータ",
                xy=[3, 6.2],
                xytext=[2, 1.5],
                arrowprops=dict(arrowstyle="->"))

graph2.annotate("理想値",
                xy=[4.4, 135],
                xytext=[3.6, 90],
                arrowprops=dict(arrowstyle="->"))
graph2.annotate("角速度",
                xy=[4, -30],
                xytext=[4, -135],
                arrowprops=dict(arrowstyle="->"))
graph2.annotate("地磁気",
                xy=[3, 10],
                xytext=[2.7, 90],
                arrowprops=dict(arrowstyle="->"))

graph3.annotate("角速度",
                xy=[4, 30],
                xytext=[4, 90],
                arrowprops=dict(arrowstyle="->"))
graph3.annotate("地磁気",
                xy=[4, -3],
                xytext=[4, -90],
                arrowprops=dict(arrowstyle="->"))

# 画像出力
fig.savefig("graph/rad/" + filename + ".png")

# plt.show()