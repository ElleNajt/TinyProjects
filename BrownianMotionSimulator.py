#Brownian Motion Simulator
#Simulate first on $R^1$
import numpy as np
import numpy
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def graph(points):
	
	data = np.array(points)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(data[:,0],data[:,1])
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	plt.axis('scaled')
	plt.show()

	return

def run1d(n):
	function = BMinterval(n)
	pointlist = convert(function)
	graph(pointlist)
	return 

def grapher(func):
	a = convert(func)
	graph(a)
	return
	
def convert(func):
	a = []
	for term in func:
		a.append([term, func[term]])
	return a
	
def BMinterval(n,startpos,starttime):
	#creates a 1-d Brownian motion on $[0,1]$ with $D_n$ dyadic level.
	
	B = {}
	i = 1
	B[starttime] = startpos
	B[starttime + 1] = startpos + np.random.randn()

	while i < n:
		k = 0
		while 2*k + 1 <= np.power(2,i):
			diadic = float(np.power(2,i))
			d = (2*k + 1) / diadic
			B[starttime + d] = startpos + .5 * ( B[starttime + (d - 1 / diadic)] + B[starttime + (d + 1 / diadic)] - 2*startpos) + .5 * np.random.randn()/ diadic
			k = k + 1
		i = i + 1
	
	return B

def BM(n,t):
	#creates a depth n brownian motion on [0,t], where t is an integer:
	i = 1
	B = BMinterval(n,0,0)
	while i < t:
		B = dict(B.items() + BMinterval(n, B[i], i).items())
		i = i + 1
	return B
	
def BM2d(n,t):
	B1 = BM(n,t)
	B2 = BM(n,t)
	list = []
	for term in B1:
		list.append([B1[term],B2[term]])
	return list
	
def fracpart(number):
	return number - np.floor(number)
	
def inbox(points):
	##Returns percentage of in points that are (up to ZxZ) in a the box, [0,1/2]x[0,1/2]
	c = 0
	for term in points:
		if (fracpart(term[0]) <= .5) and (fracpart(term[1]) <= .5):
			c = c + 1
	return c / float(len(points))

def fold(points):
	new = []
	for term in points:
		new.append([fracpart(term[0]),fracpart(term[1])])
	return new
	
def inint(points,k):
	c = 0
	for term in points:
		if (fracpart(points[term])) <= k:
			c = c + 1
	return c / float(len(points))
	
def fold2(points):
	new = []
	for term in points:
		new.append([fracpart(points[term])])
	return new