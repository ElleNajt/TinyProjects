# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:41:46 2020

@author: lnajt
"""


import networkx as nx
import copy
import random
import numpy as np
import math
#import mpmath
import time
import matplotlib.pyplot as plt
import gc
from heapq import nsmallest
from scipy import sparse

from EnumeratingConnectedPartitions import viz



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

def fast_connectivity(incidence_matrix):
    # Writing my own connectivity check , given the incidence matrix.
    # The one built into networkx seems to require instantiation of a graph
    # That is slow. This one uses numerical linear algebra instead.
    
    laplacian = np.matmul(incidence_matrix.A, incidence_matrix.A.transpose())
    spectral_gap = nsmallest(2,np.linalg.eigvals(laplacian))[-1]
    #eigval, eigvec = sparse.linalg.eigsh(laplacian, 2, sigma=0, which='LM') 
    
    if spectral_gap > .0001:
        # Replace with correct lower bound later
        return True #, eigvec
    else:
        return False #, eigvec
    
def component(graph, component_superset, node, edge_subset):
    # Return the set of nodes of graph that are in the same connected component
    # as node , using edges edge_subset. Given color_component as extra information
    # which is a set of nodes containing the component.
    
    known_component = [node]
    frontier = [node]
    while frontier:
        new_frontier = []
        remaining_nodes = []
        for z in component_superset:
            for w in frontier:
                if [z,w] or [w,z] in edge_subset:
                    known_component.append(z)
                    new_frontier.append(z)
                else:
                    remaining_nodes.append(z)
        frontier = new_frontier
        component_superset = remaining_nodes
    return known_component

def coloring_split(graph, non_cut_edges, edge):

    #This removes edge from the coloring -- if that was the bridge between the component of that coloring, it assigns an unused color to one of the components.

    coloring = graph.graph["coloring"]
    a = coloring[edge[0]]
    b = coloring[edge[1]]
    #We know a == b...
    if a != b:
        print("something went wrong!")
        
    color_component = [x for x in graph.nodes() if coloring[x] == a]

    color_subgraph_edge_list = []
    for f in non_cut_edges:
        # Note that edge is not in this set
        if coloring[f[1]] == a:
            color_subgraph_edge_list.append(f)

    # sub_incidence_matrix = nx.linalg.graphmatrix.incidence_matrix(graph, nodelist = color_component, edgelist = color_subgraph_edge_list, oriented = True)
    #We pull out the submatrix of the incidence matrix corresponding to these edges and columns.
    
    # is_connected = fast_connectivity(sub_incidence_matrix)
    b_component = component(graph, color_component, edge[1], color_subgraph_edge_list)
        
    if edge[0] in b_component:
        #This was the case that e was not a bridge
        return graph

    # Now, in the case that e was a bridge, we need to reassign the colors. 
    # First we find a color not used by the other components.

    possible_colors = set(graph.nodes())
    for used_color in coloring.values():
        if used_color in possible_colors:
            possible_colors.remove(used_color)

    unused_color = list(possible_colors)[0]


    #We let component with edge[0] keep its color, and reassign compoennt with edge[1]
    

    for y in b_component:
        coloring[y] = unused_color

    
    graph.graph["coloring"] = coloring
    return graph



def contradictions(graph, cut_edges):

    coloring = graph.graph["coloring"]
    # We use the coloring as a dynamically preprocessed calculation
    # of the span of the set of cut edges.
    
    contradictions = 0
    for e in cut_edges:
        if coloring[e[0]] == coloring[e[1]]:
            contradictions += 1
            
            
    
    return contradictions



def step(graph, cut_edges, non_cut_edges, temperature, fugacity):

    coin = random.uniform(0,1)


    if coin < 1/2:
        return non_cut_edges, cut_edges

    current_contradictions = contradictions(graph, cut_edges)
    current_edge_cuts = len(cut_edges)
    
    e = random.choice(graph.graph["ordered_edges"])

    if e in non_cut_edges:
        add_edge = False
        non_cut_edges.remove(e)
        cut_edges.add(e)
        
        graph = coloring_split(graph, non_cut_edges, e)
        #This is the update coloring that potentially reassigns the colors, because when making e cut some new components might emerge...
    else:
        add_edge = True
        non_cut_edges.add(e)
        cut_edges.remove(e)
        
        graph = update_coloring(graph, e)
        #This is the update coloring that merges the two colors, based on moving e into non-cut.

    new_contradictions = contradictions(graph, cut_edges)
    new_edge_cuts = len(cut_edges)

    if new_contradictions <= current_contradictions:
        return non_cut_edges, cut_edges

    coin = random.uniform(0,1)

    if (temperature**(new_contradictions - current_contradictions))* (fugacity**(new_edge_cuts - current_edge_cuts)) <= coin:        
        if add_edge == True:
            non_cut_edges.remove(e)
            cut_edges.add(e)
            graph = coloring_split(graph, non_cut_edges, e)
        if add_edge == False:
            non_cut_edges.add(e)
            cut_edges.remove(e)
            graph = update_coloring(graph, e)
        return non_cut_edges, cut_edges
    else:
        return non_cut_edges, cut_edges


def run_markov_chain(graph, steps = 100, temperature = .5, fugacity = 1):
    #Return a sample as [Cutedges, non_cutedges, coloring]

    #temperature = .8
    #steps = 1000
    graph.graph["ordered_edges"] = list(graph.edges())
    graph.graph["coloring"] = {x : x for x in graph.nodes()}

    cut_edges = set( graph.graph["ordered_edges"])
    non_cut_edges = set([])

    sample = False
    num_found = 0
    for i in range(steps):
        non_cut_edges, cut_edges = step(graph, cut_edges, non_cut_edges, temperature, fugacity)
        if contradictions(graph, cut_edges) == 0:
            num_found += 1
            sample = [copy.deepcopy(cut_edges),copy.deepcopy(non_cut_edges), copy.deepcopy(graph.graph["coloring"])]

    print("found this many samples:" , num_found)

    return sample


def test_grid_graph():

    #for fugacity in [.2,.4,.5,.6,.8,1,2,3,4,5]:
    for fugacity in [1]:
        size = 10
        graph = nx.grid_graph([size, size])
        MC_steps = 10000000
        MC_temperature = .8
        print(str(fugacity))
        sample = run_markov_chain(graph, MC_steps, MC_temperature, fugacity)
        name = "markov_chain" + str(fugacity)
        print(name)
        if sample != False:
            viz(graph, sample[1], sample[2], name)
        else:
            print("no sample found")
            
test_grid_graph()

#started 7:11

#Under 4 minutes.