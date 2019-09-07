import numpy as np

def loss(x, t):
    return np.average((x - t)**2)
    

x = np.array([1,2,3])
t = np.array([4,5,6])
print(x)
print(t)
print(loss(x, t))