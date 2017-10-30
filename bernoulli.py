##Bernoulli shift numerics
#list = iterate(4,[[0,.5]])

import numpy as np

def iterate(n,set):
	setlist = [set]
	i = 0
	while i < n:
		set = mix(set)
		setlist.append(set)
		i = i + 1
	return setlist
	
def mix(set):
	#returns $(x2)^{-1}$ of the set
	#stored as a list of [[a1,b1],[a2,b2]...], meaning [a1, b1] cup [a2,b2]...
	newset = []
	for term in set:
		new1 = [term[0]/2, term[1]/2]
		new2 = [.5 + term[0]/2, .5 + term[1]/2]
		newset.append(new1)
		newset.append(new2)
	return newset

def function(list,t):
	i = 0
	sum = 0
	n = len(list)
	while i < n:
		for term in list[i]:
			if (term[0] <= t) and (t <= term[1]):
				sum = sum + 1
		i = i + 1
	return sum / (float(n))

#function(iterate(5,[[0,.5]]),0)

def preparegrapher(n):
	list = iterate(n,[[.12,.52]])
	xvalues = numpy.random.uniform(0,1,5000)
	yvalues = [function(list,x) for x in xvalues]
	return [xvalues,yvalues]
	
		
import matplotlib as mpl
import numpy
import matplotlib.pyplot as plt

def stats(list,epsilon = .01):
	c = 0
	for term in list:
		if np.abs(term[1] - .4) <= epsilon:
			c = c + 1
	return c / float(len(list))

def graph(n):
	list = preparegrapher(n)
	plt.scatter(list[0],list[1])
	plt.show()
	return list
	
def show(set):
	points = []
	dumb = []
	for item in set:
		points.append(np.random.uniform(item[0],item[1],100))
		dumb.append(np.random.uniform(0,0,100))
	plt.scatter(dumb, points[:])
	plt.show()
	return points