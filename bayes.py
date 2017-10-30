#!/usr/bin/python

######Arff Reading

######------------------ Arff reader

def loaddata(filename):
	f = open(filename, 'r')
	lines = f.readlines()
	i = 0
	while lines[i][0:9].upper() != '@RELATION':
		i = i + 1
	relation = lines[i][10:]
	attributes = {}
	#attributes are ordered
	j = 0
	ordering = {}
	while lines[i][0:5].upper() != '@DATA':
		if lines[i][0:10].upper() == '@ATTRIBUTE':
			if "{" in lines[i]:
				temp = lines[i].split("{")
				attributes[temp[0].split(" ")[1]] = [x.strip() for x in temp[1].split("}")[0].split(",")]
				ordering[j] = temp[0].split(" ")[1]
				j = j +1
			#Remove whitespaces later...
			#   else:
			#    Numeric data or other...			
		i = i + 1
	
	i = i + 1
	#Now reading data
	data = []
	while i < len(lines):
		if ((lines[i].split(",")[0]) in attributes[ordering[0]]) & (lines[i][0] != '#'):
			data.append(lines[i].split(","))
		i = i + 1

	return [attributes, ordering, data]
	
#################GRAPHS:
###Some ideas here learned from : https://www.python.org/doc/essays/graphs/

#I define a weighted direct graph as a dictionary, where the assignment to each node is the list of edges going out along with their weights

#For example: graph2 = {'A': [['B',1], ['D',10]],'B': [['C',1], ['D',10]],'C': [['D',1]],'D': [['E',3]],'E' : []}

#Be sure to include even the nodes without any exiting arrows... otherwise the tree from prims will not span...

#whenconstructing graph from .arff, make sure that the input ordering maps monotonically onto the ordering in nodes(graph)


######------------------ Construction of mutual information graph:


import math
def initializegraph(classprob, classcounter, attributecounter, attributes, ordering, data, Laplace):
	i = 0
	j = 0
	graph = {}
	while i < len(ordering) - 1:
		j = 0
		listtempforgraph = []
		while j < len(ordering) - 1:
			listtempforgraph.append([j, mutualinfo(i,j, attributes, ordering, data, attributecounter, Laplace)])
			j = j + 1
		graph[i] = listtempforgraph
		i = i + 1

	return graph
	

#########PRIMS:


def weight(path):
	i = 0
	for node in path:
		i = i + node[1]
	return i

def treeweight(tree):
	i = 0
	for node in tree:
		for edge in tree[node]:
			i = i + edge[1]
	return i

def nodes(path):
	list = []
	for stop in path:
		list.append(stop[0])
	return list

def find_shortest_path(graph, start, end):
	return call_find_shortest_path(graph,[start,0],end)

def call_find_shortest_path(graph,start,end,path=[]):		
	path = path + [start]
	if start[0] == end:
		return path
	if not graph.has_key(start[0]):
		return None
	shortest = None
	for node in graph[start[0]]:
		if node[0] not in nodes(path):
			newpath = call_find_shortest_path(graph, node, end, path)
			if newpath:
				if not shortest or weight(newpath) < weight(shortest):
					shortest = newpath
	return shortest

def nodes2(graph):
	list = []
	for stop in graph:
		list.append(stop)
	return list

def treenodes(tree):
	list = []
	for entry in tree:
		list.append(entry[1][0])
	return list

def firstindex(list, input):
	if input not in list:
		return None
	i = 0
	while i < len(list):
		if list[i] == input:
			return i
		i = i + 1
graph = {0 : [[0,0]]}
def prims_algorithm(graph, tree = [[graph.items()[0][0],[graph.items()[0][0],0]]]):	
	bestedge = None
	maxweight = -10000
	
	for node in treenodes(tree):		
		for edge in graph[node]:
			if (edge[1] > maxweight) and (edge[0] not in treenodes(tree)):
				maxweight = edge[1]
				bestedge = [node,[edge[0],edge[1]]]

			if (edge[1] == maxweight) & (edge[0] not in nodes(tree)):
				#tie breaking - 
				if firstindex(nodes2(graph), node) < firstindex(nodes2(graph), bestedge[0]):
					bestedge = [node,[edge[0],edge[1]]]
				if (firstindex(nodes2(graph), node) == firstindex(nodes2(graph), bestedge[0])) and (firstindex(nodes2(graph), edge[0]) < firstindex(nodes2(graph), bestedge[1][0])):
					bestedge = [node,[edge[0],edge[1]]]
					
	tree = tree + [bestedge]
	
	if len(tree) < len(nodes2(graph)):
		tree = prims_algorithm(graph, tree)
	#print(treeweight(tree))
	return (tree)

def fixtree(tree):
	dict = {}
	for term in tree:
		dict[term[0]] = []
	for term in tree:
		dict[term[0]] = dict[term[0]] + [term[1]]
	return dict

def parents(node, graph):
	listparents = []
	for item in graph:
		if node in [x[0] for x in graph[item]]:
			listparents.append(item)
	return listparents

#############

def addclass(tree, ordering):
	tree[len(ordering) -1] = []
	for item in ordering:
		if item != len(ordering) -1:
			tree[len(ordering) - 1] = tree[len(ordering) - 1] + [[item,0]]
	return tree	
	
def cleantree(tree):
	list = tree[0]
	list.pop(0)
	##Note this pops from within the tree already... why?
	return tree
	
def removeclass(tree, ordering):
	tree[len(ordering) - 1] = []
	tree.pop(len(ordering) - 1)
	return tree
	
def children(n, tree):
	if n not in tree.keys():
		return "none"
	if tree[n] == []:
		return "none"
	if n in tree.keys():
		return [x[0] for x in tree[n]]
	

def leaves(tree, ordering):
	leaflist = []
	
	#Don't call ordering -- makes a bad algorithm
	
	
	
	for node in ordering:
		if children(node, tree) == "none":
			leaflist.append(node)
	node.remove(len(orderin) - 1)
	#Because of the way ordering is constructed...
	return leaflist
	
def calllevellist(tree, ordering):
	tree2 = copy.deepcopy(tree)
	return levellist(tree2,ordering)
	
def levellist(tree, ordering, ongoinglist = []):
	if len(tree) == 0:
		return []
	leaflist = leaves(tree, ordering)  #Need to pop them from ordering also... so leaves is a bad algorithm, rewrite it
	
	for node in tree:
	
	
	
		ongoinglist = ongoinglist + [leaflist]
		print("NEXT LEVEL")
		print(leaflist)
		print(tree)
		for leaf in leaflist:
			for node in tree:
				for x in tree[node]:
					if x[0] == leaf:
						tree[node].remove(x)
		print(tree)
		for key in tree.keys():
			if tree[key] == []:
				tree.pop(key)
		print(tree)
	return levellist(tree, ordering, ongoinglist)				
					
					
def testlistmake(n,list = ["start"]):
	list = list + [n]
	n = n - 1
	if n == 0:
		return list
	return testlistmake(n,list)
	

def treerevtopordering(tree, ordering):
	import copy
	#create new ordering of ordering, corresponding to revtop order of tree
	tree = removeclass(tree, ordering)
	tree2 = copy.deepcopy(tree)
	
def parent(n, tree):
	for node in tree:
		if n in children(node,tree):
			return node
	return "root"
		
	
################PROBABILITY:


#####Marginals

def condprob(n, feature, y, attributecounter):
	#This computes P(x_n = feature | y)
	#Verified by hand
	K = 0
	for term in attributecounter:
		if (term[1].strip() == y) & (term[0][0] == n) & (term[0][1].strip() == feature):
			K = attributecounter[term] + K
			
	N = 0
	for term in attributecounter:
		if (term[1].strip() == y) & (term[0][0] == n):
			N = attributecounter[term] + N
	return float(K) / float(N)
	
def condprobdata(i, feature_i, y, data,attributes, ordering, Laplace):
	TotalCount = len(attributes[ordering[i]])
	K = Laplace
	N = Laplace*TotalCount
	for term in data:
		if (term[i] == feature_i) & (term[len(data[0])- 1].strip() == y):
			K = K + 1
			
	for term in data:
		if term[len(data[0]) - 1].strip() == y:
			N = N + 1
	return float(K) / float(N)

def classprobs1(i, y, data,attributes, ordering, Laplace):
	TotalCount = len(attributes[ordering[i]])
	K = Laplace
	N = Laplace*TotalCount
	for term in data:
		if term[len(data[0]) - 1].strip() == y:
			K = K + 1
	for term in data:
		N = N + 1
	return float(K) / float(N)
	
def classprobs2(i, j, y, data,attributes, ordering, Laplace):
	TotalCount = len(attributes[ordering[i]])*len(attributes[ordering[j]])
	K = Laplace
	N = Laplace*TotalCount
	
	for term in data:
		if (term[len(data[0])- 1].strip() == y):
			K = K + 1
			
	for term in data:
		N = N + 1

	return float(K) / float(N)
	
def condpairprob(i, feature_i, j, feature_j, y, data,attributes, ordering, Laplace):
	#This computes P(x_i = feature_i, x_j = feature_j | y)

	#Verified with:
	#data2 = [[1,1,1],[1,1,1],[1,0,0],[1,0,1]]
	#And variations on that
	#data3 = [ [str(x) for x in y] for y in data2]
	#condpairprob(0,"1",1,"1","1",data3)
	TotalCount = len(attributes[ordering[i]])*len(attributes[ordering[j]])
	K = Laplace
	N = Laplace*TotalCount
	
	for term in data:
		if (term[i] == feature_i) & (term[j] == feature_j) & (term[len(data[0])- 1].strip() == y):
			K = K + 1
			
	for term in data:
		if term[len(data[0]) - 1].strip() == y:
			N = N + 1

	return float(K) / float(N)
	
def condon2(i, feature_i, j, feature_j, y, data,attributes, ordering, Laplace):	
	#To compute P(x_i | x_j, y)
	TotalCount = len(attributes[ordering[i]])
	K = Laplace
	N = Laplace*TotalCount

	for term in data:
		if (term[i] == feature_i) & (term[j] == feature_j) & (term[len(data[0])- 1].strip() == y):
			K = K + 1
			
	for term in data:
		if (term[j] == feature_j) & (term[len(data[0]) - 1].strip() == y):
			N = N + 1

	return float(K) / float(N)
	
	
def condpairprob2(i, feature_i, j, feature_j, data,attributes, ordering, Laplace):
	#This computes P(x_i = feature_i, x_j = feature_j)
	#Note that because we are using pseudo counts, we need to run these seperately.
	#
	TotalCount = len(attributes[ordering[i]])*len(attributes[ordering[j]])
	K = Laplace
	N = Laplace*TotalCount
	for term in data:
		if (term[i] == feature_i) & (term[j] == feature_j):
			K = K + 1
	N = N + len(data)
	return float(K) / float(N)
	
	
def condtripleprob(i, feature_i, j, feature_j, y, data, attributes, ordering, Laplace):
	TotalCount = len(attributes[ordering[i]])*len(attributes[ordering[j]])*len(attributes[ordering[len(data[0])-1]])
	K = Laplace
	N = Laplace*TotalCount
	for term in data:
		if (term[i] == feature_i) & (term[j] == feature_j) & (term[len(data[0])- 1].strip() == y):
			K = K + 1	
	N = N + len(data)
	return float(K) / float(N)

#condtripleprob(1,attributes[ordering[10]][0],3,attributes[ordering[3]][0],attributes[ordering[len(ordering) - 1]][0],data)	
#condpairprob2(1,attributes[ordering[1]][0],3,attributes[ordering[3]][0],data)
#condprobdata(1,attributes[ordering[1]][0],attributes[ordering[len(ordering) - 1]][0],data)	
###########################


def naivebayes(y, xvalues, attributecounter, classprob, ordering, attributes):

	#This returns P(Y = y | xvalues) 
	#If no evidence for xn, put NOEVIDENCE
	y = y.strip()
	k = 0
	N = float(classprob[y])
	while k < (len(ordering) - 1):
		if xvalues[k] != "NOEVIDENCE":
			N = N * condprob(k,xvalues[k],y,attributecounter)
		k = k + 1

	D = float(0)
	for classterm in attributes[ordering[len(ordering) - 1]]:
		M = float(classprob[classterm])
		k = 0
		while k < (len(ordering) - 1):
			if xvalues[k] != "NOEVIDENCE":
				M = M * condprob(k,xvalues[k],classterm, attributecounter)
			k = k + 1
		D = D + M

	if D != 0:
		return N / D

	return N/D
	
###############



def tanbayes(y, xvalues, attributecounter, classprob, ordering, attributes, data, tree, Laplace):
	y = y.strip()
	
	#Part One: For each node in tree, compute P(nodevalue, parentvalue)
	#Need to call: Parent(n,tree), and access counts.
	#condpairprob(10,data[0][10], 11, data[0][11], data[0][len(data[0])-1].strip(), data)
	#Note that we have to strip the second to last entry
	
	ordering2 = copy.deepcopy(ordering)
	ordering2.pop(len(ordering)-1)
	P = float(1)
	for term in ordering2:
		parentalfigure = parent(term,tree)
		if parentalfigure == "root":
			P = P*condprobdata(term, xvalues[term], y, data,attributes, ordering, Laplace)
			#P has been multiplied by prob(term, y)
		if parentalfigure != "root":
			P = P*condon2(term, xvalues[term], parentalfigure, xvalues[parentalfigure], y, data,attributes, ordering, Laplace)
			#P has been multipliedby P(term | y,parentalfigure) -- but check
	
	P = P*classprob[y]
	return P

def normalizedtanbayes(y, xvalues, attributecounter, classprob, ordering, attributes, data, tree,Laplace):
	y = y.strip()
	N = tanbayes(y, xvalues, attributecounter, classprob, ordering, attributes, data, tree, Laplace)*100000000000
	D = float(0)
	for y in attributes[ordering[len(ordering) - 1]]:
		D = D + tanbayes(y, xvalues, attributecounter, classprob, ordering, attributes, data, tree, Laplace)*100000000000
	return float(N) / float(D)
	
def classifytan(xvalues, attributecounter, classprob, ordering, attributes, data, tree, Laplace):	
	scores = {}
	for y in attributes[ordering[len(ordering) - 1]]:
		scores[y] = normalizedtanbayes(y, xvalues, attributecounter, classprob, ordering, attributes, data, tree, Laplace)
	max = 0
	for key in scores.keys():
		if scores[key] >= max:
			maxclass = key
			max = scores[key]
	list = [maxclass, max]
	return list
	
def classifynaive(xvalues, attributecounter, classprob, ordering, attributes):
	scores = {}
	for y in attributes[ordering[len(ordering) - 1]]:
		scores[y] = naivebayes(y, xvalues, attributecounter, classprob, ordering, attributes)
		#Sanity adds to one
	#N = 0
	#for key in scores.keys():
	#	N = N + scores[key]
	#print N
	max = 0
	for key in scores.keys():
		if scores[key] >= max:
			maxclass = key
			max = scores[key]
	list = [maxclass, max]
	return list
	
###############TESTS:

	
	
	
	
#tanbayes("metastases",testdata[k][0:len(data[0]) -1], attributecounter, classprob, ordering, attributes, data, tree)

#tanbayes("malign_lymph",testdata[k][0:len(data[0]) -1], attributecounter, classprob, ordering, attributes, data, tree)

####################MUTUAL INFORMATION:




def mutualtest(i,j):
	return mutualinfo(i, j, attributes, ordering, data, attributecounter, Laplace)

######
#i = 0
#j = 0
#N = 0 
#while i < len(data[0])- 1:
#	while j < len(data[0]) - 1:
#		if (mutualtest(i,j)) <= 0.03:
#			if mutualtest(i,j) >= 0.025:
#				print (i + "   "  + j + mututaltest(i,j))
##		j = j + 1
#	i = i + 1
	
def mutualinfo(i, j, attributes, ordering, data, attributecounter, Laplace):
	if i == j:
		return -1
		
	#Check on mutualtest(i,j) = mutualtest(j,i) <- This works
	SUM = 0
	for classterm in attributes[ordering[len(ordering) -1]]:
		y = classterm
		N = 0	
		for termi in attributes[ordering[i]]:
			for termj in attributes[ordering[j]]:		
				M = float(condtripleprob(i, termi, j, termj, y, data,attributes, ordering, Laplace))
				M2i = float(condprobdata(i, termi, y, data,attributes, ordering, Laplace))
				M2j = float(condprobdata(j, termj, y, data,attributes, ordering, Laplace))
				M2ij = float(condpairprob(i, termi, j, termj, y, data,attributes, ordering, Laplace))
				if (M2ij != 0) & (M2j != 0) & (M2i != 0) :
					M3 = math.log(M2ij/(M2i*M2j),2)
					MM = M * M3
					N = N + MM					
		SUM = SUM + N
	return SUM
###### POPULATOR


#####---- Then populate
###### Populate counters:


def populator(ordering, attributes, data, Laplace):

	i = len(ordering) - 1 
	classcounter = {}
	attributecounter = {}

	for term in attributes[ordering[i]]:
		
		classcounter[term] = Laplace

	for classterm in attributes[ordering[i]]:
		k = 0
		while k < i:
			for attributeterm in attributes[ordering[k]]:
				attributecounter[ ((k, attributeterm), classterm)] = Laplace
			k = k + 1

	j = 0
	while j < len(data):
		for classterm in attributes[ordering[i]]:
			if data[j][i].strip() == classterm.strip():
				classcounter[classterm] = classcounter[classterm] + 1
			k = 0
			while k < i:
				for attributeterm in attributes[ordering[k]]:
					if data[j][i].strip() == classterm.strip():
						if data[j][k].strip() == attributeterm.strip():
							attributecounter[ ((k, attributeterm), classterm)] = attributecounter[ ((k, attributeterm), classterm)] + 1
				k = k + 1		
		j = j + 1
		
	Y = 0
	for term in classcounter:
		Y = classcounter[term] + Y

	classprob = {}

	for term in attributes[ordering[i]]:
		classprob[term] = float(classcounter[term]) / float(Y)
		
	return [classcounter, classprob, attributecounter]

############------------------

def partialcounter(y,n):
	N = 0
	for term in attributecounter:
		if (term[1].strip() == y) & (term[0][0] == n):
			N = attributecounter[term] + N
	return N
	
#######################BAYES:


#####Main Executable

## Load Main Data

trainsetfile = "lymph_train.arff.txt"
testsetfile = "lymph_test.arff"
#bayes(trainsetfile, testsetfile, "t")
import math
import random
import copy
import sys

def bayes(trainsetfile, testsetfile, type, samplesize = -1):
	Laplace = 1
	#Load training data
	#list = loaddata('lymph_train.arff.txt')
	
	
	list = loaddata(trainsetfile)
	attributes = list[0]
	ordering = list[1]
	data= list[2]
	
	if samplesize != -1:
		data = random.sample(data, samplesize)
		
	list = populator(ordering, attributes, data, Laplace)
	classcounter = list[0]
	classprob = list[1]
	attributecounter = list[2]

	if type == "t":
		####Tan specialized algorithms
		graph = initializegraph(classprob, classcounter, attributecounter, attributes, ordering, data, Laplace)
		tree = fixtree(prims_algorithm(graph))

		tree = cleantree(tree)
	#Loads test data:
	#testlist = loaddata('lymph_test.arff')
	
	testlist = loaddata(testsetfile)
	testattributes = testlist[0]
	testordering = testlist[1]
	testdata = testlist[2]

	testlist = populator(ordering, attributes, testdata, Laplace)
	testclasscounter = testlist[0]
	testclassprob = testlist[1]
	testattributecounter = testlist[2]

	

	if type == "n":
	###########NAIVE BAYES RUN:
		for numb in ordering:
			if numb != len(ordering) -1:
				print (ordering[numb].replace("'","") + " " + ordering[len(ordering) -1].replace("'",""))
		print("")
		k = 0
		correct = 0
		while k < len(testdata) :
			list = classifynaive(testdata[k][0:len(data[0]) -1], attributecounter, classprob, testordering, attributes)
			print(list[0] + " " + testdata[k][len(data[0]) - 1].strip() + " " + str(list[1]))
			if list[0]  == testdata[k][len(data[0]) - 1].strip():
				correct = correct + 1
			k = k + 1
		print("")
		print correct
		
	#print tree
	if type == "t":
		###########TAN BAYES RUN:
		for node in ordering:
			if node != len(ordering) - 1:
				parentalfigure = parent(node,tree)
				if parentalfigure != "root":
					print(ordering[node].replace("'","") + " " + ordering[parentalfigure].replace("'","") + " " + ordering[len(ordering) -1].replace("'",""))
				if parentalfigure == "root":
					print(ordering[node].replace("'","") + " " + ordering[len(ordering) -1].replace("'",""))
			
		print("")
		k = 0
		correct = 0
		while k < len(testdata) :
			list = classifytan(testdata[k][0:len(data[0]) -1], attributecounter, classprob, ordering, attributes, data, tree, Laplace)

			print(list[0] + " " + testdata[k][len(data[0]) - 1].strip() + " " + str(list[1]))

			if list[0]  == testdata[k][len(data[0]) - 1].strip():
				correct = correct + 1
			k = k + 1

		
		print("")
		print correct

#####

bayes(sys.argv[1], sys.argv[2], sys.argv[3])
	