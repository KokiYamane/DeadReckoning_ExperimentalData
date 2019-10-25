import numpy as np
import matplotlib.pyplot as plt

# データ読み込み関数
def loaddata(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1,
                      unpack=False, dtype=str)
    speed = []
    xyzwave = []
    for row in data:
        speed.append(row[1].astype('f8'))
        xyzwave.append(row[2: 2 + 3 * 50].astype('f8'))
    speed = np.array(speed)
    x = np.array(xyzwave)
    t = speed[:, np.newaxis]
    return x, t

# データ読み込み関数
def loadrtkdata(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1,
                      unpack=False, dtype=str)
    latitude = []
    longitude = []
    for row in data:
        latitude.append(row[2].astype('f8'))
        longitude.append(row[3].astype('f8'))
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    return latitude, longitude

# 5[Hz] から 1[Hz] に
def changeHz(inputlist):
    returnlist = []
    param = 5  # [Hz]
    for i in range(int(len(inputlist)/param)):
        if i < param:
            continue
        partialSum = 0
        for j in range(param):
            partialSum += inputlist[param*i-j]
        returnlist.append(partialSum / param)
    return returnlist

# データ読み込み
filename = '1010'
acc, speed = loaddata('ML/' + filename + '.csv')
latitude, longitude = loadrtkdata('rtk/csv/' + filename + 'rtk.csv')

# グラフ表示
fig = plt.figure(figsize=(8, 6))

graph1 = fig.add_subplot(121)
graph1.plot(changeHz(latitude), changeHz(longitude))
graph1.set_aspect('equal')
graph1.set_xlabel('latitude')
graph1.set_ylabel('longitud')
graph1.get_xaxis().get_major_formatter().set_useOffset(False)
graph1.get_yaxis().get_major_formatter().set_useOffset(False)
graph1.grid()

graph2_1 = fig.add_subplot(122)
graph2_1.set_xlabel('time [s]')
graph2_1.plot(changeHz(longitude), label='longitude')
graph2_1.get_yaxis().get_major_formatter().set_useOffset(False)
graph2_1.grid()

graph2_2 = graph2_1.twinx()
graph2_2.set_ylabel('speed [m/s]')
graph2_2.plot(speed, color='gray', label='speed')

h1, l1 = graph2_1.get_legend_handles_labels()
h2, l2 = graph2_2.get_legend_handles_labels()
graph2_1.legend(h1+h2, l1+l2, loc='lower right')
graph2_1.tick_params(labelleft=False, left=False)
graph1.tick_params(labelleft=False, labelbottom=False,
                   left=False, bottom=False)
plt.subplots_adjust(left=-0.1, wspace=-0.2)

plt.savefig('graph/' + filename + 'speed.png')


minute = 5
x = changeHz(longitude[0:5*60 * minute])
y = changeHz(latitude[0:5*60 * minute])

fig, axes = plt.subplots(figsize=(10, 5))
axes.scatter(x, y, marker='.')
axes.set_aspect('equal')
plt.xlabel('longitud')
plt.ylabel('latitude')
xmargin = (max(x) - min(x)) * 0.05
plt.xlim(min(x) - xmargin, max(x) + xmargin)
ymargin = (max(y) - min(y)) * 0.05
plt.ylim(min(y) - ymargin, max(y) + ymargin)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
plt.grid()
plt.savefig('graph/' + filename + 'route.png')

plt.show()