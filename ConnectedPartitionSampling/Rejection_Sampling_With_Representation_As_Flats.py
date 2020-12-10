# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 05:08:31 2020

@author: lnajt

Uses rejection sampling to produce connected partitions of a graph.
We represent connected partitions by the set of edges internal to each block.
In matroid terminology, this is the flat associated to the connected partition.

We rejection sample by picking a random subset of edges and checking whether 
it is a flat.


"""

import networkx as nx
import copy
import random

def rejection_sample(graph):
    """
    

    Parameters
    ----------
    graph : networkx graph
        the graph we want a connected partition of.

    Returns
    -------
    False if it didn't produce a connected partition, otherwise:
        
    a list containing the edge set of the connected partition, and a 
    dictionary representing one possible corresponding node coloring

    """

    
    J = []
    I = []
    for e in graph.edges():
        c = random.uniform(0,1)
        if c > .5 :
            J.append(e)
        else:
            I.append(e)

    graph.graph["coloring"] = {x : x for x in graph.nodes()}

    for e in J:
        graph = update_coloring(graph, e)

    coloring = graph.graph["coloring"]
    for e in I:
        if coloring[e[0]] == coloring[e[1]]:
            return False
    return [J, coloring]

def update_coloring(graph, edge):
    #Here we add edge to the set of non-cut edges, meaning that the two endpoints are now definately in the same connected compoennt. WE update the coloring to reflect that.
    coloring = graph.graph["coloring"]
    a = coloring[edge[0]]
    b = coloring[edge[1]]
    if a == b:
        return graph
        #If already in the same component, do nothing.
    else:
        #Update the component coloring by the smallest label.
        #TODO: I think there's a way to speed this up, by redefining "b" to be "a" in some way (or vica versa)
        if a < b:
            for x in graph.nodes():
                if coloring[x] == b:
                    coloring[x] = a
        else:
            for x in graph.nodes():
                if coloring[x] == a:
                    coloring[x] = b
    return graph

def test_rejection_sample(graph, goal = 1):
    """
    keeps attempting to rejection sample connected partitions of G
    until it produces 'goal' many connected partitions
    """
    samples = []
    sample_colors = []
    number = 0

    while number < goal:
        new = rejection_sample(graph)

        if new != False:
            number += 1
            samples.append(new)
            sample_colors.append(graph.graph["coloring"])

    print("Got ", number, " samples")

    return samples

def estimate_ratio(graph, trials = 100000):
    """
    to estimate the rate at which we can obtain connected partitions
    """
    number = 0
    for i in range(trials):
        new = rejection_sample(graph)
        if new != False:
            number += 1
    return number, trials, number/trials


input_graph = nx.grid_graph([3,3,3])
sample = test_rejection_sample(input_graph,1)

print(sample[0][0])
print(estimate_ratio(input_graph, 10000))