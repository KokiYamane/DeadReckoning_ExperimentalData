import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Yu Gothic"]

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

for i in range(3):
    time,acc_x,acc_y,acc_z,vel_x,vel_y,vel_z,pos_x,pos_y,pos_z = np.loadtxt(str(i) + ".txt", delimiter = ",", skiprows = 1, unpack = True)

    fig1 = plt.figure(figsize=(6, 6))

    graph = fig1.add_subplot(311)
    # graph.plot(time,acc_x, color="red", label = "x")
    # graph.plot(time,acc_y, color="blue", label = "y")
    graph.plot(time-4,acc_z, color="gray", label = "raw data")
    graph.plot(time-4,LPF(acc_z), color="black", label = "LPF")
    plt.tick_params(labelbottom=False, bottom=False)
    graph.set_ylabel("加速度 [m/s^2]")
    graph.yaxis.set_label_coords(-0.1, 0.5)
    plt.xlim(0,30)
    plt.ylim(-0.15,0.15)
    # plt.legend(bbox_to_anchor=(0, 1), loc="upper left", borderaxespad=0, fontsize=12)
    plt.grid()

    # 注釈
    graph.annotate("生データ",
                    xy=[20, 0.01],
                    xytext=[20, 0.1],
                    arrowprops=dict(arrowstyle="->"))
    graph.annotate("LPF後のデータ",
                    xy=[20, 0],
                    xytext=[20, -0.1],
                    arrowprops=dict(arrowstyle="->"))

    graph = fig1.add_subplot(312)
    # graph.plot(time,vel_x, color="red")
    # graph.plot(time,vel_y, color="blue")
    graph.plot(time-4,vel_z, color="black")
    plt.tick_params(labelbottom=False, bottom=False)
    graph.set_ylabel("速度 [m/s]")
    graph.yaxis.set_label_coords(-0.1, 0.5)
    # plt.xlim(4,34)
    plt.xlim(0,30)
    plt.ylim(-0.025,0.1)
    plt.grid()

    graph = fig1.add_subplot(313)
    # graph.plot(time,pos_x, color="red")
    # graph.plot(time,pos_y, color="blue")
    graph.plot(time-4,pos_z, color="black")
    # plt.xlim(4,34)
    plt.xlim(0,30)
    plt.ylim(-0.5,1.5)
    graph.set_ylabel("位置 [m]")
    graph.yaxis.set_label_coords(-0.1, 0.5)
    plt.grid()

    graph.set_xlabel("時間 [秒]")

    plt.savefig(str(i) + ".png")
