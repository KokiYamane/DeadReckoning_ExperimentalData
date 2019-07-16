import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

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

# データの読み込み
filename = "0716acc"
data = np.loadtxt("data/" + filename + ".csv", delimiter = ",", skiprows = 2, unpack = True) 

fig = plt.figure(figsize=(6, 6))
graph = fig.add_subplot(111)

graph.plot(data[item.pos_step_gyro_x],data[item.pos_step_gyro_y], color="red", label="gyro")
graph.plot(data[item.pos_step_mag_x],data[item.pos_step_mag_y], color="blue", label="mag")
# graph.plot(data[item.pos_step_x],data[item.pos_step_y], color="gray")
# graph.plot(data[item.pos_x],data[item.pos_y], color="blue", label="mag")

graph.set_xlabel("x [m]")
graph.set_ylabel("y [m]")

graph.set_aspect("equal")

plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0, fontsize=12)

plt.savefig("graph/route/" + filename + ".png")

plt.show()
