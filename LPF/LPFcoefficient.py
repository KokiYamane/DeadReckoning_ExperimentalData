import numpy as np
import matplotlib.pyplot as plt

data = [0.1]
for i in range(1,100):
    data.append(0.9 * data[i-1])

fig = plt.figure(figsize=(6,6))
graph = fig.add_subplot(111)
graph.plot(data)
plt.xlabel("number")
plt.ylabel("coefficient")
plt.savefig("graph/LPFcoefficient.png")

# DFT後のデータ
f = open("LPFcoefficient.txt", mode="wt")
for i in data:
    f.write("{:8f}".format(i) + "\n")

data2 = np.loadtxt("LPFcoefficient_DFT.txt", delimiter = "\t", unpack = True) 
fig2 = plt.figure(figsize=(6,6))
graph2 = fig2.add_subplot(111)
f = np.arange(0.0, 50.0, 0.5)
graph2.plot(f,data2[0])
plt.xlabel("frequency [Hz]")
plt.ylabel("amplitude [dB]")
plt.savefig("graph/LPFcoefficient_DFT.png")
# plt.show()