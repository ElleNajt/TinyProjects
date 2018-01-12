import numpy as np
from scipy import spatial
import scipy
import copy
samplesize = 100
xvals = np.random.rand(samplesize,2) - .5
yvals = []
for x in xvals:
	yvals.append(np.sign(x[0] + x[1]))
	
#yvals += np.random.normal(0,.001,samplesize)

wrecord = []
w = np.random.rand(2) - .5
wrecord.append(copy.deepcopy(w))
mu = .01
d = 1000
while d >= 1:
	i = np.random.choice(range(len(xvals)))
	x = xvals[i]
	y = yvals[i]
	yhat = np.sign(np.dot(w,x))
	w += mu* ( y - yhat)* x
	predictions = np.sign(np.dot(xvals, w))
	d = scipy.spatial.distance.hamming(predictions, yvals)*len(yvals)
	print(d)
	wrecord.append(copy.deepcopy(w))
	
##visualization:

from matplotlib import animation
from matplotlib import pyplot as plt

fig = plt.figure()
ax = plt.axes(xlim=(-1,1),ylim = (-1,1))
line, = ax.plot([],[],lw=2)

xminus = []
xplus = []
i = 0
while i < len(xvals):
	if yvals[i] == 1:
		xplus.append(xvals[i])
	if yvals[i] == -1:
		xminus.append(xvals[i])
	i += 1

A = np.transpose(xminus)
B = np.transpose(xplus)
plt.scatter(A[0], A[1])
plt.scatter(B[0], B[1])

def init():
	line.set_data([],[])
	return line,
	
def animate(i):
	t = np.linspace(-1,1,1000)
	w = wrecord[i % len(wrecord)]
	if w[1] != 0:
		x = t
		y = -1 * (w[0] / w[1]) * x
	else:
		y = t
		x = -1 * (w[1] / w[0]) * y
	line.set_data(x,y)
	return line,
	
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(wrecord), interval=20, repeat_delay = 20000, blit=True)
plt.show()