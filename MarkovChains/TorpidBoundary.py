# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:05:20 2019

@author: Temporary
"""

from Torpid_Mixing import random_walk
from Torpid_Mixing import animate
import numpy as np
import networkx as nx

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
    
    for x in boundary.nodes():
        boundary.node[x]["score"] = 0
    for x in boundary.nodes():
        if x[0] == 0 or x[0] == m - 1:
            boundary.node[x]["score"] += 1
    for x in boundary.nodes():
        if x[1] == 0 or x[1] == m - 1:
            boundary.node[x]["score"] += -1
    #To test an make sure it assigned the scores correctly:
#    values = [boundary.node[x]["score"] for x in boundary.nodes()]
#    nx.draw(boundary, pos=nx.get_node_attributes(boundary, 'pos'), node_size = 10, width = .5, cmap=plt.get_cmap('jet'), node_color=values)
                        


def evaluate_block(boundary, block):
    total = 0
    for x in block:
        total += boundary.node[x]["score"]
    return total
    
def create_time_series(boundary, restricted_path):
    series = []
    for block in restricted_path:
        series.append(evaluate_block(boundary, block))
    return series

###############################################

##WorkFlowHere:
    
def entire_workflow(m, steps):
    grid = nx.grid_graph([m,m])
    for v in grid.nodes():
        grid.node[v]["X"]= v[0]
        grid.node[v]["Y"]= v[1]
    grid.graph["size"] = 0
    grid.graph["steps"] = steps
    path = random_walk(grid, steps,False)
    
    
    
    boundary_nodes = []
    
    for x in grid.nodes():
        if grid.degree(x) <= 3:
            boundary_nodes.append(x)
    
    boundary = nx.subgraph(grid, boundary_nodes)
    
    restricted_path = restrict_path(boundary_nodes, path)
    
    
    assign_weights(boundary) 
    
    series = create_time_series(boundary, restricted_path)
    return series

m = 10
steps = 2000

series = entire_workflow(m,steps)

print(series)
print(np.mean(series))


