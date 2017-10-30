################################################
#############ARFF READER:

#!/usr/bin/python

import sys

######Arff Reading
###SPEAK:  [attributes, ordering, data] = loaddata("sonar.arff.txt")
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
	classtypes = 0
	ordering = {}
	while lines[i][0:5].upper() != '@DATA':
		if lines[i][0:10].upper() == '@ATTRIBUTE':
			###FOr nominal inputs, has { }...
			if "{" in lines[i]:
				temp = lines[i].split("{")
				attributes[temp[0].split(" ")[1]] = [x.strip() for x in temp[1].split("}")[0].split(",")]
				ordering[j] = temp[0].split(" ")[1]
				j = j +1
			else:
				temporary = lines[i]
				temp = temporary.split(" ")
				attributes[temp[1]] = temp[2]
				ordering[j] = temp[1]
				j = j +1
				
			#Remove whitespaces later...
			#   else:
			#    Numeric data or other...
		
		i = i + 1
	
	i = i + 1
	#Now reading data
	data = []
	while i < len(lines):
		#if ((lines[i].split(",")[0]) in attributes[ordering[0]]) & (lines[i][0] != '#'):
		data.append(lines[i].split(","))
		i = i + 1

	return [attributes, ordering, data]
	
#######################################################################
################INITIALIZER:

#Neural Net initializer
import math
import random

#Im storing a bipartite graph as a dictionary. 

def buildflow(left, right):
	#This will take two sets of nodes, and build from the first to the second all edges, and store this as a directed graph. So we pass the attributes + bias, hiddenattributes first, then hiddenattributes + hiddenbias, output.
	
	graph = {}
	
	for term in left:
		list = []
		for target in right:
			x = random.uniform(0, .1)
			list.append([ str(target), x])
		graph[term] = list
		
	return graph
	
def connect(firstpart, secondpart):
	#Takes two bipartite graphs, with output ofthe first the same  as the input of the secnod, and links them into a big graph. 
	
	#We don't make any next connections here, the point is that we store the outputs of the first using the same names (as strings) that we store the inputs ofthe second graph with

	graph = copy.deepcopy(firstpart)
	for term in secondpart:
		graph[term] = secondpart[term]
		
	return graph
	
#dict = {"'Class'" : 1, "A" : 1, "B" : 2, "C" : 3}
#net2 = buildnetwork(dict)

def buildnetwork(attributes):

	attributeback = copy.deepcopy(attributes)
	output = attributeback.pop("'Class'")
	attributelist = []
	hiddenattributelist = []
	for term in attributeback:
		attributelist.append(term)
		hiddenattributelist.append("h" + term)
		
	attributelist.append("bias")
	firstpart = buildflow(attributelist, hiddenattributelist)
	hiddenattributelist.append("hbias")
	secondpart = buildflow(hiddenattributelist, ["Class"])
	
	#Fix so have one bias unit -- it will connect to the hidden layer and also class
	net = connect(firstpart, secondpart)
	biasterm = net["bias"] + net["hbias"]
	net["bias"] = biasterm
	net.pop("hbias")
	
	net["'Class'"] = []
	return net
	
#############################################################
############BACKPROPEGATION

#Backpropegation Computation

#Start with a bipartite graph of the correct form, and a piece of data
#Forward propegate through network, and compute error. Then use backprop to compute gradients

#Check backprop formulas...





def computegradient(attributes, net, ordering, datapoint):
	#Forward propegate:
	activation = push(net, ordering, datapoint)
	delweights = errorpush(net,ordering, activation, attributes)
	return delweights
	
def errorpush(net, ordering, activation, attributes):
	list = hpusherror(net, ordering, activation, attributes)
	delweights = featurepusherror(net, ordering, activation, attributes, list[0], list[1])
	return delweights
	
def featurepusherror(net, ordering, activation, attributes, error, delweights):
	for term in ordering:
		name = ordering[term]
		if name != "'Class'":
			for hterm in net[name]:
				hname = hterm[0]
				delweights[name + hname] = activation[name] * error[hname]
	
	name = "bias"
	for term in ordering:
		hname = ordering[term]
		if hname != "'Class'":
				hname = "h" + hname
				delweights[name + hname] = error[hname]
				
	return delweights

def hpusherror(net, ordering, activation, attributes):
	#Takes in a forward propegated net
	error = {}
	delweights = {}
	outputerror = errorcalc(net, attributes, activation)[0]
	error["'Class'"] = outputerror
	for term in ordering:
		name = ordering[term]
		if name != "'Class'":
			name = "h" + name
			weightot = net[name][0][1]
			A = weightot*outputerror
			list = net[name]
			activa = activation[name]
			termerror = A * activa * ( 1 - activa)
			error[name] = termerror
			DelWclassterm = outputerror * activation[name]
			delweights[name +  "'Class'"] = DelWclassterm
			
	name = "bias"
	delweights[name +"'Class'"] = outputerror
	
	####Why do bias weights not matter???
	
	return [error, delweights]
	
def errorcalc(network, attributes, activation, threshold = .5):
	trueclass = activation["TrueClass"]
	predictor = activation["'Class'"]
	if predictor < threshold:
		predictedclass = attributes["'Class'"][0]
		if predictedclass != trueclass:
			error = predictor - 1
		if predictedclass == trueclass:
			error = predictor
	if predictor >= threshold:
		predictedclass = attributes["'Class'"][1]
		if predictedclass != trueclass:
			error = predictor
		if predictedclass == trueclass:
			error = predictor - 1
		
	#This seems like a reasonable way to do error -- distance of sigmoid output from true value, 0 or 1.... check arithmetic...
	return [error, predictedclass]
	
###########################
###########FORWARD PROPEGATION:

#Forward Propegation Algorithm

def sigmoid(x):
	#Maybe there is better way in floating point to do this computation?
	try:
		y = float(1) / (float(1) + math.exp(-1 * x))
	except OverflowError:
		#Since overflow happens if exp "=" infty..., i.e x is very big an negative... but how coudl this happen
		y = 0
	return y
	
def hpush(net, ordering, activation):
	#Takes a net and actiation, and pushes the activation forward to the hidden layer
	i = 0
	while i < len(ordering) - 1:
		sum = float(0)
		j = 0
		while j < len(ordering) - 1:
			xjwi = activation[ordering[j]]*calculateweight(ordering[j], "h" + ordering[i], net)
			sum = sum + xjwi
			j = j + 1
		sum = sum + calculateweight("bias", "h" + ordering[i], net)
		activation["h" + ordering[i]] = sigmoid(sum)
		i = i + 1
	
	return activation

def classpush(net, ordering, activation):
	#Extends the activation function to the class
	i = 0
	sum = 0
	while i < len(ordering) - 1:
		xiwclass = activation["h" + ordering[i]]*calculateweight("h" + ordering[i], 'Class', net)
		sum = sum + xiwclass
		i = i + 1
	sum = sum + calculateweight("bias", 'Class', net)
	activation["'Class'"] = sigmoid(sum)
	return activation
		
def value(list):
	return(list[len(list) - 1])

def calculateweight(parent, child, net):
	list = net[parent]
	for term in list:
		if type(term) is type([]):
			if term[0]  == child:
				return float(term[1])
			
	##If parent does not link, then adds nothing. Justifies the zero below:
	return 0
	
#net2 = prep(net,ordering, data[0])
def activate(net, ordering, datapoint):
	#defines the initial values of the actiation function
	i = 0
	activation = {}
	#net2 = copy.deepcopy(net)
	while i < len(ordering) - 1:
		activation[ordering[i]] = float(datapoint[i])
		i = i + 1
	activation["TrueClass"] = datapoint[i].strip()
	return activation
	
def push(net, ordering, datapoint):
	activation = activate(net, ordering, datapoint)
	activation = hpush(net, ordering, activation)
	activation = classpush(net, ordering, activation)
	
	return activation
	
#
####################################################################
###########GRADIENT DESCENT

#Gradient Descent

def update(net, gradient, rate, ordering):

	#I think the reason why this is so slow is because I am deleting, re creating and inserting big chunks from the 

	#hidden layer to class
	for term in ordering:
		if ordering[term] != "'Class'":
			name = "h" + ordering[term]
			weight = net[name][0][1]
			weight = weight - rate*gradient[name + "'Class'"]
			list2 = [[net[name][0][0], weight]]
			net[name] = list2
				
			
	#Bias node to class
	name = "bias"
	list = net[name]
	k = 0
	while k < len(list):
		if list[k][0] == 'Class':
			weight = list[k][1]
			list.pop(k)
		k = k + 1
	weight = weight - rate*gradient[name + "'Class'"]
	term = ['Class', weight]
	list.append(term)
	net[name] = list

	#Features to hidden layer
	
	for term in ordering:
		if ordering[term] != "'Class'":
			name = ordering[term]
			k = 0
			while k < len(net[name]):
				hterm = net[name][k]
				hname = hterm[0]
				weight = hterm[1]
				weight = weight - rate*gradient[name + hname]
				list = net[name]
				list = replace(list, k, [hname, weight])
				net[name] = list
				k = k + 1
				
				
	#Update bias to hfeatures
	
	k = 0
	while k < len(net["bias"]):
		if (net["bias"][k][0] != 'Class'):
			hname = net["bias"][k][0]
			weight = net["bias"][k][1]
			weight = weight - rate * gradient["bias" + hname]
			list = net["bias"]
			list = replace(list, k, [hname, weight])
			net["bias"] = list
		k = k + 1
	
	return net
	
def replace(list, k, new):
	#Replaces the kth spot of list with new
	list1 = list[0:k+1]
	list2 = list[k+1:len(list)]
	list1.pop(k)
	list1.append(new)
	final = list1 + list2
	return final
	
################################################################################
##################MAIN

#net = buildnetwork(attributes)
#delta = computegradient(attributes, net, ordering, data[10])
#net2 = update(net, delta, 1)
import copy
import math


def learnnet(filename, epochs):
	#list = loaddata("OUTPUTINDICATOR")
	list = loaddata(filename)
	#list = loaddata("TESTINDICATOR")
	attributes = list[0]
	ordering = list[1]
	data = list[2]
	net = buildnetwork(attributes)
	k = 0
	
	while k < epochs:
		#activation = push(net, ordering, data[2])
		#print errorcalc(net, attributes, activation)
		#activation = push(net, ordering, data[1])
		#print errorcalc(net, attributes, activation)
		print k
		net = iterate(net, data, attributes, ordering,1,.1)
		k = k + 1
	
	return net

def iterate(net, data, attributes, ordering, batchsize, learningrate):
	random.shuffle(data)
	#randomly re-arrange data?
	k = 0
	while k < len(data):
		i = 0
		sumdelta = 0

		while i < batchsize:
			delta = computegradient(attributes, net, ordering, data[k])
			#Need command to add functions on weights... or to add two dictionaries with common domain... and need to define the zero dictionary to initialize 
			#sumdelta = sumdelta + delta
			i = i + 1
			k = k + 1
		net = update(net, delta, learningrate, ordering)

	return net
	
def predict(net, data, atributes, ordering, datapoint):
	activation = push(net, ordering, datapoint)
	return errorcalc(net, attributes, activation)
	
def getpredictions(net, data, attributes, ordering):
	for term in data:
		error = predict(net, data, attributes, ordering, term)
		print("Error, predicted class:" + str(error))
	return error
	
##################################################################################
############################CROSS VALIDATION;

#Cross Validation

def partition(data, n):
	#if n doesn't divide len(data), the put extra terms on tail of list of lists
	m = len(data) /n
	listoflists = []
	i = 0
	while i < n - 1:
		listoflists.append(data[i*m: (i+1)*m ])
		i = i + 1
	listoflists.append(data[i*m : len(data)])
	return listoflists

def crossvalidate(filename, folds, rate, epochs):
	list = loaddata(filename)
	#list = loaddata("sonar.arff.txt")

	attributes = list[0]
	ordering = list[1]
	data = list[2]
	net = buildnetwork(attributes)
	#randomly re-order data
	#random.shuffle(data)

	###Statified Cross Validation
	i = 0
	positives = []
	negatives = []
	while i < len(data):
		if (data[i][len(data[i]) - 1]).strip() == attributes["'Class'"][1]:
			positives.append(i)
		else:
			negatives.append(i)
		i = i + 1
	positiveslist = partition(positives, folds)
	negativeslist = partition(negatives, folds)

	k = 0
	totalerror = 0
	totalwrong = 0
	listofoutputs = []
	while k < len(positiveslist):
		#k is the fold number
		#print("Fold " + str(k))
		validationdata = []
		traindata = []
		i = 0
		while i < len(data):
			if (i in positiveslist[k]) or (i in negativeslist[k]):
				validationdata.append(data[i])
			else:
				traindata.append(data[i])
			i = i + 1

		valuesfromfold = runfold(k,traindata, validationdata, attributes, ordering, net, rate, epochs)
		listofoutputs = listofoutputs + (valuesfromfold[6])
		errorinfold = valuesfromfold[0]
		wronginfold = valuesfromfold[1]
		totalerror = totalerror + errorinfold
		totalwrong = totalwrong + wronginfold
		k = k + 1
	error = float(totalerror) / float(folds + 1)
	#print("average error across all folds was : " + str(error))
	#print("error overall was: " + str( float(totalwrong) / float(len(data))))
	correctedorder = ordercorrector(listofoutputs, positiveslist, negativeslist)
	#this is necessary because our stratified parittion disorded the elements, according to positiveslist and negativeslist... so the above function computes the permutation from original order, to order in listofoutputs, compute the inverse of that permutation, and then applies it...
	for term in correctedorder:
		print term
	return correctedorder
	
def ordercorrector(list, positiveslist, negativeslist):
	#Note that from the list of positives and negatives, we can figure out the original permutation. 
	#It is (sort(positiveslist[0], negativeslist[0]), sort(positiveslist[1], negativeslist[1]), ...) so we compute this element of the symmetric group, and then compute its inverse
	
	sigma = []
	k = 0
	while k < len(positiveslist):
		templist = positiveslist[k] + negativeslist[k]
		templist.sort()
		sigma = sigma + templist
		k = k + 1
	#using all of these templist, list2, etc. variable names seems like bad practice, because it leads to programming errors, but I don'tknow a good alternative
	
	#this has computed the permutation
	
	list2 = reorder(list, invert(sigma))
	return list2
	
	
def invert(sigma):
	#inverts a symmetric group element
	i = 0
	inverse = {}
	while i < len(sigma):
		inverse[sigma[i]] = i
		i = i + 1
	inverselist = []
	for term in inverse:
		inverselist.append(inverse[term])
		
	return inverse
	
def reorder(list, permutation):
	#takes in a permutation and a list of things, and apply the permutation to the list.
	correctedlist = []
	
	for term in permutation:
		correctedlist.append(list[permutation[term]])
		
	
	return correctedlist

def runfold(foldnumber, traindata, validationdata, attributes, ordering, net, rate, epochs):
	net = trainfold(traindata, attributes, ordering, net, rate, epochs)
	return validatefold(foldnumber, validationdata, attributes, ordering, net,1,.5)
	
def validatefold(foldnumber, validationdata, attributes, ordering, net, printlist, threshold):
	k = 0
	wrong = 0
	falsenegative = 0
	falsepositive = 0
	truenegative = 0
	truepositive = 0
	listofoutputs = []
	#Am considering the first class listed negative (0), the second positive (1)
	#In mine vs rock case, a falsenegative is not identifying mine, a truepositive is identifying a mine.
	while k < len(validationdata):
		activation = push(net, ordering, validationdata[k])
		errorinfo = errorcalc(net, attributes, activation, threshold)
		if (printlist == 1) or (printlist == 2):
			listofoutputs.append( (str(foldnumber) + " " + str(errorinfo[1]) + " " + activation["TrueClass"] + " " + str(activation["'Class'"])))
		if errorinfo[1] != activation["TrueClass"]:
			if activation["TrueClass"] == attributes["'Class'"][1]:
				falsenegative = falsenegative + 1
			if activation["TrueClass"] == attributes["'Class'"][0]:
				falsepositive = falsepositive + 1
			wrong = wrong + 1
		if errorinfo[1] == activation["TrueClass"]:
			if activation["TrueClass"] == attributes["'Class'"][1]:
				truepositive = truepositive + 1
			if activation["TrueClass"] == attributes["'Class'"][0]:
				truenegative = truenegative + 1
			
		k= k + 1
	error = float(wrong) / float(len(validationdata))
	if printlist == 2:
		print("error in this fold was : " + str(error) + "and size of validation set was : " + str(len(validationdata)))
	return [error, wrong, falsenegative, falsepositive, truepositive, truenegative, listofoutputs]

def trainfold(traindata, attributes, ordering, net, rate, epochs):
	k = 0
	while k < epochs:
		#print (str(k) + "th epoch")
		net = iterate(net, traindata, attributes, ordering,1,rate)
		k = k + 1
	
	return net

##############################################################################
####ROC CURVES

def roc(net, validationdata, attributes, ordering, scale = .01):
	threshold = 0
	posdata = []
	while threshold <= 1:
		newlist = validatefold(0, validationdata, attributes, ordering, net, 0, threshold)
		posdata.append([newlist[2], newlist[3], newlist[4], newlist[5], threshold])
		threshold = threshold + scale
	return posdata

#import matplotlib.pyplot as plt

#[error, wrong, falsenegative, falsepositive, truepositive, truenegative]
#               positive, negative, positive, negative

def prepareFPRvsTPR(net, validationdata, attributes, ordering):
	list = roc(net, validationdata, attributes, ordering, .1)
	FPRvsTPR = []
	for term in list:
		P = term[0] + term[2]
		N = term[1] + term[3]
		
		if P == 0:
			TPR = 0
		else:
			TPR = float(term[2]) / float(P)
			
		if N == 0:
			FPR = 0
		else:
			FPR = float(term[1]) / float(N)
		FPRvsTPR.append([FPR, TPR, term[4]])
	
	return FPRvsTPR

#list = prepareFPRvsTPR(net, validationdata, attributes, ordering)	
		
		
		
		
		
	
#########################################################
#############EXTRA GOODY: CREATING TOY DATA FOR DEBUGGING

#Create sample arff files for simple binary classification with numerical features... for debugging purposes.

#Learn indicator -- given a rectangle in the unit cube R^5, we output some M random points in the unit cube, and classify them based on whether tor not they are are in the rectangle:

#This samples from the indicator of [0,1/2]^n

def uniformdraw(dimension, number):

	pointlist = []
	i = 0
	while i < number:
		newpoint = []
		k = 0
		while k < dimension:
			newpoint.append(random.uniform(0,1))
			k = k + 1
		newpoint.append(inrectangle(newpoint))
		pointlist.append(newpoint)
		i = i + 1
	
	return pointlist
	
def inrectangle(list):
	for term in list:
		if term > .5:
			return 0
	return 1
	
def writetofile(list, dimension):
	#f = open("TESTINDICATOR",'w')
	f.write("@relation indicator \n")
	i = 0
	while i < dimension:
		f.write("@attribute x_" + str(i) + " numeric \n")
		i = i + 1
	f.write("@attribute 'Class' {0,1} \n")
	f.write("@data \n")
	for term in list:
		for subterm in term:
			f.write(str(subterm) + ",")
		f.write("\n")
	
###############################################
####Stats -- 

def getstatistics(type, folds, epochs, last):
	
	#list = loaddata(filename)
	list = loaddata("sonar.arff.txt")
	listofroc = []
	rate = .1
	attributes = list[0]
	ordering = list[1]
	data = list[2]
	net = buildnetwork(attributes)
	#randomly re-order data
	#random.shuffle(data)
	###Statified Cross Validation
	i = 0
	positives = []
	negatives = []
	while i < len(data):
		if (data[i][len(data[i]) - 1]).strip() == attributes["'Class'"][1]:
			positives.append(i)
		else:
			negatives.append(i)
		i = i + 1
	positiveslist = partition(positives, folds)
	negativeslist = partition(negatives, folds)

	k = 0
	totalerror = 0
	totalwrong = 0
	listofoutputs = []
	while k < len(positiveslist):
		#k is the fold number

		validationdata = []
		traindata = []
		i = 0
		while i < len(data):
			if (i in positiveslist[k]) or (i in negativeslist[k]):
				validationdata.append(data[i])
			else:
				traindata.append(data[i])
			i = i + 1

		nettrained = trainfold(traindata, attributes, ordering, net, rate, epochs)
		list = prepareFPRvsTPR(nettrained, validationdata, attributes, ordering)
		listofroc.append(list)
		
		k = k + 1
	
	g = open("ROCDATA", 'w')
	g.write(str(listofroc))
	
	return listofroc

def averagelistoflist(list):
	#Take a list of lists of vectors, and average them
	d = len(list)
	averagelist = list.pop(0)

	for term in list:
		k = 0
		while k < len(term):
			averagelist[k] = addvector(averagelist[k], term[k])
			k = k + 1
	
	j = 0 
	while j < len(averagelist):
		new = dividevectorbyscalar(averagelist[j], d)
		averagelist = replace(averagelist, j, new)
		j = j + 1
	
	return averagelist
	
def replace(list, k, new):
	#Replaces the kth spot of list with new
	list1 = list[0:k+1]
	list2 = list[k+1:len(list)]
	list1.pop(k)
	list1.append(new)
	final = list1 + list2
	return final
	
def addvector(list1, list2):
	list = []
	k = 0
	while k < len(list1):
		term = list1[k] + list2[k]
		list.append(term)
		k =k + 1
	return list
			
def dividevectorbyscalar(enterlist, d):
	list = []
	for term in enterlist:
		list.append(term / float(d))
	return list

############################################################################################

#This is the code for classification of the dataset
crossvalidate(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))

#This is the code for producing statistics:
#getstatistics(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))




