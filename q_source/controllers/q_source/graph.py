import matplotlib.pyplot as plt
import numpy as np

#x1, y1 = np.loadtxt('reward.txt', unpack=True)
#plt.plot(x1,y1, label='ro')

#plt.scatter(x,y, label='skitscat', color='k', s=25, marker="o")

plt.xlabel('Iteration number')
plt.ylabel('Rewards earned')
plt.title('Iterations vs Rewards')
plt.legend()
plt.show()
fig,ax=plt.subplots()
x1, y1 = np.loadtxt('reward1.txt',unpack=True)
ax.plot(x1,y1,color='b', label='robot 1')

x2, y2 = np.loadtxt('reward2.txt',unpack=True)
ax.plot(x2,y2,color='g', label='robot 2')

x3, y3 = np.loadtxt('reward3.txt',unpack=True)
ax.plot(x3,y3,color='r', label='robot 3')

x4, y4 = np.loadtxt('reward4.txt',unpack=True)
ax.plot(x4,y4,color='y', label='robot 4')

x5, y5 = np.loadtxt('reward5.txt',unpack=True)
ax.plot(x5,y5,color='c', label='robot 5')
#plt.scatter(x,y, label='skitscat', color='k', s=25, marker="o")

ax.xlabel('Iteration number')
ax.ylabel('sum val earned')
plt.title('Iterations vs qsum')
ax.legend()
plt.show()