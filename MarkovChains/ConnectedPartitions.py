# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Lorenzo Najt
"""

import networkx as nx
import random
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import math


def get_blocks(graph):
    #Returns the blocks induced by "assignment"
    #You can do this in O(n), not O(n^2)
    block_table = []
    for c in graph.graph["colors"]:
        block = []
        for x in graph.nodes():
            if graph.graph["assignment"][x] == c:
                block.append(x)
        block_table.append(block)
    return block_table

def check_connected(graph):
    #Checks that the blocks induced by "assignment" give connected subgraphs
    #This can be optimized a ton
    block_table = get_blocks(graph)
    for block in block_table:
        subgraph = nx.subgraph(graph, block)
        if len(block) != 0:
            if nx.is_connected(subgraph) == False:
                return False
    return True

def step(graph):
    #Markov chain step
    n= len(graph.nodes())
    old_number_colors = len ( set ( graph.graph["assignment"].values()))
    
    graph.graph["num_colors_history"].append(old_number_colors)
    for x in graph.nodes():
        graph.graph["memory"][x] += graph.graph["assignment"][x]
    x = random.choice(list(graph.nodes()))
    c = random.choice(graph.graph["colors"])
    old_color = graph.graph["assignment"][x]
    graph.graph["assignment"][x] = c
    if not check_connected(graph):
        graph.graph["assignment"][x] = old_color
        return False
    new_number_colors = len ( set ( graph.graph["assignment"].values()))
    
    
    ##Metropolis Hastings to prevent over counting of stationary distribution  due to labelling
    #That is : note that a partition into n isolated blocks has n! representations, but one into a single block has n representations.
    if new_number_colors <= old_number_colors:
        return True
    
    c = random.uniform(0,1)

    if c < score(n,new_number_colors)/score(n,old_number_colors):
        return True
    else:
        graph.graph["assignment"][x] = old_color
        return False


def score(n, k):
    # A partition with k current colors on n nodes, is represented
    # (n choose k) k! = n^{falling(k)} times. 
    # up to a constant, this is is 1 / (n - k)! times. 
    
    #So we want to weight a partition by (n - k)!. 
    
    return math.factorial(n - k)
    
    
    
def initialize(size):
    graph = nx.grid_2d_graph(size,size)
    
    colors = range(2*len(graph.nodes()))
    #reasonable idea : add a pool of 2n colors, so there's always something to swap in. This does change the necessary reweightings.
    
    graph.graph["colors"] = colors
    
    assignment = {}
    
    memory = {}
    average_color = {}
    for x in graph.nodes():
        assignment[x] = 0
        memory[x] = 0
        average_color[x] = 0
        
    graph.graph["assignment"] = assignment
    graph.graph["memory"]= memory
    graph.graph["average_color"] = average_color
    graph.graph["num_colors_history"] = []
    return graph

def reset_memory(graph):
    #Resets the average color memory
    memory = {}
    for x in graph.nodes():
        memory[x] = 0

    graph.graph["memory"]= memory

def viz(graph):
    '''
    Draws the graph , labelling nodes either by the current assignment ("assignment") or the average assignment ("display_memory")
    '''
    for x in graph.nodes():
        graph.node[x]["pos"] = [x[0], x[1]]
    for x in graph.nodes():
        graph.node[x]["col"] = graph.graph["assignment"][x] 
    values = [graph.node[x]["col"] for x in graph.nodes()]
    
    fig = plt.figure()
    gs = gridspec.GridSpec(2,2)
    
    ax = plt.subplot(gs[0,0])
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'),labels = graph.graph["average_color"], node_size = 10, width = .5, cmap=plt.get_cmap('jet'), node_color=values)
    fig.add_subplot(ax)
    ax = plt.subplot(gs[0,1])
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'),labels = graph.graph["assignment"], node_size = 10, width = .5, cmap=plt.get_cmap('jet'), node_color=values)
    fig.add_subplot(ax)
    ax = plt.subplot(gs[1,0])
    plt.plot(list(range(len(graph.graph["num_colors_history"]))), graph.graph["num_colors_history"])
    fig.add_subplot(ax)
    
size = 6
graph = initialize(size)
steps = 1000
for i in range(steps):
    step(graph)
    
reset_memory(graph)

for i in range(steps):
    step(graph)
    

for x in graph.nodes():
    graph.graph["average_color"][x] = int(graph.graph["memory"][x]/ steps)
viz(graph)

    
    
