# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:05:20 2019

@author: Lorenzo Najt
"""

from Torpid_Mixing import random_walk
from Torpid_Mixing import animate
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def viz(T,k,n):
    for x in T.nodes():
        T.node[x]["pos"] = [T.node[x]["X"], T.node[x]["Y"]]
    for x in T.nodes():
        if x in n:
            T.node[x]["col"] = 1
        else:
            T.node[x]["col"] = 0
    values = [T.node[x]["col"] for x in T.nodes()]
    nx.draw(T, pos=nx.get_node_attributes(T, 'pos'), node_size = 200/k, width = .5, cmap=plt.get_cmap('jet'), node_color=values)
    
#viz(grid, 10,path[5])




def restriction(boundary_nodes, block):
    restricted_block = []
    for x in block:
        if x in boundary_nodes:
            restricted_block.append(x)
    return frozenset(restricted_block)

def restrict_path(boundary_nodes, path):
    restricted_path = []
    for block in path:
        restricted_path.append(restriction(boundary_nodes, block))
        
    return restricted_path

#viz(boundary, m, restricted_path[5])

####Now we are going to add a function to the vertice of boundary, which evaluates by summing over the vertices in the block. We design it to have symmetries so that a the expected score (if the block is drawnfrom the pushforwardof uniform) is zero.

def assign_weights(boundary):
    m = (len(boundary.nodes()) + 4) / 4
    for x in boundary.nodes():
        boundary.node[x]["score"] = 0
    for x in boundary.nodes():
        if x[0] == 0 or x[0] == m - 1:
            boundary.node[x]["score"] += 1
    for x in boundary.nodes():
        if x[1] == 0 or x[1] == m - 1:
            boundary.node[x]["score"] += -1
            
def assign_complex_weights(boundary):
    m = (len(boundary.nodes()) + 4) / 4
    for x in boundary.nodes():
        boundary.node[x]["score"] = 0
    for x in boundary.nodes():
        if x[0] == 0:
            boundary.node[x]["score"] += 1           
        if x[0] == m - 1:
            boundary.node[x]["score"] += -1
    for x in boundary.nodes():
        if x[1] == 0:
            boundary.node[x]["score"] += 0+1j
        if x[1] == m - 1:
            boundary.node[x]["score"] += 0-1j
    #To test an make sure it assigned the scores correctly:
#    
#    
#values = [boundary.node[x]["score"].imag for x in boundary.nodes()]
#nx.draw(boundary, pos=nx.get_node_attributes(boundary, 'pos'), node_size = 10, width = .5, cmap=plt.get_cmap('jet'), node_color=values)
#                

def assign_smooth_weights(boundary):
    m = (len(boundary.nodes()) + 4) / 4
    for x in boundary.nodes():
        boundary.node[x]["score"] = 0
    for x in boundary.nodes():
        if x[0] == 0 or x[0] == m - 1:
            boundary.node[x]["score"] += np.exp( - (1/m)*( x[1] - m/2)**2)
    for x in boundary.nodes():
        if x[1] == 0 or x[1] == m - 1:
            boundary.node[x]["score"] += -1 *  np.exp( - (1/m)*( x[0] - m/2)**2)
            
def assign_smooth_complex_weights(boundary):
    m = (len(boundary.nodes()) + 4) / 4
    for x in boundary.nodes():
        boundary.node[x]["score"] = 0
    for x in boundary.nodes():
        if x[0] == 0:
            boundary.node[x]["score"] += 1*np.exp( - (1/m)*( x[0] - m/2)**2)      
        if x[0] == m - 1:
            boundary.node[x]["score"] += -1*np.exp( - (1/m)*( x[0] - m/2)**2)
    for x in boundary.nodes():
        if x[1] == 0:
            boundary.node[x]["score"] += (0+1j)*np.exp( - (1/m)*( x[1] - m/2)**2)
        if x[1] == m - 1:
            boundary.node[x]["score"] += (0-1j)*np.exp( - (1/m)*( x[1] - m/2)**2)
        
######################################


#Find boundary of boundary  (stored as edges)
    
def find_endpoints(boundary, block):
    #These are the endpoints of the induced SAW in the dual (or empty if SAP)
    #Note that there is asubtle bug here -- the order of the endpoints is not determined
    #uniquely as it was written originally. That is, they are organized to be continuous.
    endpoints = []
    for x in block:
        for y in boundary.neighbors(x):
            if y not in block:
                endpoints.append([x,y])
    #NB: endpoints might be empty -- this is the case of the trivial partition. 

    return endpoints

### To make endpoints outputs continuous:
    #TODO -- this still won't handle trivial partitions without throwing an error.
    
def contiguify(series):
    ordered_series = []
    for i in range(len(series)):
        if i == 0:
            current_a = np.array(series[i][0])
            current_b = np.array(series[i][1])
        if i > 0:
            if np.linalg.norm( np.array(series[i][1]) - current_a) <= 1:
                current_a = np.array(series[i][1])
                current_b = np.array(series[i][0])
            else:
                current_a = np.array(series[i][0])
                current_b = np.array(series[i][1])
        ordered_series.append( [list(current_a), list(current_b)])
    return ordered_series

def evaluate_endpoints(boundary, endpoints):
    total = 0
    for e in endpoints:
        subtotal = 0
        for y in e:
            subtotal += boundary.node[y]["score"]
        total += subtotal
#        if subtotal > 0:
#            total += 1
#        if subtotal < 0:
#            total += -1
    return total
        
def smooth_evaluate_endpoints(boundary, endpoints):
    
    total = 0
    for e in endpoints:
        for y in e:
            total += boundary.node[y]["score"]
    return total

def xycoordinates(boundary, endpoints):
    coords = []
    for e in endpoints:
        xcoord = 0
        ycoord = 0
        for y in e:
            xcoord += y[0]/2
            ycoord += y[1]/2
            
        coords.append( [ xcoord, ycoord])
    return coords
#################

def endpoint_coordinates(boundary, block):
    endpoints = find_endpoints(boundary, block)
    return xycoordinates(boundary, endpoints)

def sum_evaluate_block(boundary, block):
    total = 0
    for x in block:
        total += boundary.node[x]["score"]
    return total

def evaluate_SAW(boundary, block):
    #This one is invariant of the orientation / choice of ordering.
    endpoints = find_endpoints(boundary, block)
    return evaluate_endpoints(boundary, endpoints)

def smooth_evaluate_SAW(boundary, block):
    endpoints = find_endpoints(boundary, block)
    return smooth_evaluate_endpoints(boundary, endpoints)
    
    
def create_time_series(boundary, restricted_path, function):
    series = []
    for block in restricted_path:
        series.append(function(boundary, block))
    return series

def create_time_series_means(boundary, restricted_path, function):
    series = []
    for block in restricted_path:
        series.append(function(boundary, block))
    mean_series = [ np.mean( series[0:n]) for n in range(1,len(series))]
    return mean_series

###############################################

##WorkFlowHere:
    
def entire_workflow(m, steps, evaluate_function, weight_function):
    grid = nx.grid_graph([m,m])
    for v in grid.nodes():
        grid.node[v]["X"]= v[0]
        grid.node[v]["Y"]= v[1] 
        grid.node[v]["pos"] = [v[0], v[1]]
    grid.graph["size"] = 0
    grid.graph["steps"] = steps
    path = random_walk(grid, steps,True)
    
    
    
    boundary_nodes = []
    
    for x in grid.nodes():
        if grid.degree(x) <= 3:
            boundary_nodes.append(x)
    
    boundary = nx.subgraph(grid, boundary_nodes)
    
    restricted_path = restrict_path(boundary_nodes, path)
    
    
    weight_function(boundary)
    
    series = create_time_series(boundary, restricted_path, evaluate_function)
    return [series, restricted_path, boundary]

m = 5
steps = 10000

evaluate_function = smooth_evaluate_SAW
weight_function = assign_smooth_complex_weights
series, restricted_path, boundary = entire_workflow(m,steps, evaluate_function, weight_function)

##

#evaluate_function = smooth_evaluate_SAW
#weight_function = assign_smooth_complex_weights
#weight_function(boundary)
#series = create_time_series(boundary, restricted_path, evaluate_function)
#
#print(np.mean(series))
#
#times = list(range(len(series)))
#plt.plot(times, [x.imag for x in series])
#plt.plot(times, [x.real for x in series])
#plt.show()


evaluate_function = endpoint_coordinates
weight_function = assign_complex_weights
series = create_time_series(boundary, restricted_path, evaluate_function)
series_continuous = contiguify(series)
print(np.mean(series))

times = list(range(len(series)))
xcoord_of_a = [x[0][1] for x in series_continuous]
plt.plot(times, xcoord_of_a)
plt.plot(times, [x[0][0] for x in series_continuous])
plt.show()

#
#evaluate_function = evaluate_SAW
#weight_function = assign_complex_weights
#series = create_time_series_means(boundary, restricted_path, evaluate_function)
#print(np.mean(series))
#
#times = list(range(len(series)))
#plt.plot(times, [x.imag for x in series])
#plt.plot(times, [x.real for x in series])
#plt.show()
#
#
##
#evaluate_function = evaluate_SAW
#weight_function = assign_complex_weights
#
#weight_function(boundary)
#series = create_time_series(boundary, restricted_path, evaluate_function)
##print(series)
#print(np.mean(series))
#
#times = list(range(len(series)))
#plt.plot(times, [x.imag for x in series])
#plt.plot(times, [x.real for x in series])
#plt.show()

#The statistic used here is not invariant of the ordering.

#evaluate_function = sum_evaluate_block
#weight_function = assign_weights
#
#weight_function(boundary)
#series = create_time_series(boundary, restricted_path, evaluate_function)
#print(series)
#print(np.mean(series))
#
#times = list(range(len(series)))
#plt.plot(times, series)
#plt.show()
