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
        graph.graph["num_colors"] += -1
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
    
    known_component =set( [node])
    frontier = set([node])
    component_superset.remove(node)
    #print("in component")
    #print("frontier = ", frontier)
    #rint(" component_superset = " , component_superset)
    #print("edge_subset = " , edge_subset)
    while frontier:
        #print("frontier: ", frontier)
        #print(" super_set:" , component_superset)
        new_frontier = set()
        remaining_nodes = set()
        for z in component_superset:
            connected = False
            for w in frontier:
                if (z,w) in edge_subset or (w,z) in edge_subset:
                    known_component.add(z)
                    new_frontier.add(z)
                    connected = True
            if connected == False:
                remaining_nodes.add(z)
        frontier = new_frontier
        component_superset = remaining_nodes
    #print("out component'")
    return known_component

def test_component():
    # Code to test the custom component function
    
    graph = nx.grid_graph([5,5])
    graph = nx.erdos_renyi_graph(10,.1)
    graph_nodes = list(graph.nodes())
    graph_edges = list(graph.edges())
    node = graph_nodes[0]
    print(node)
    nx.draw(graph, with_labels = True)
    print(component(graph, graph_nodes, node, graph_edges))
    print(nx.is_connected(graph))

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
    graph.graph["num_colors"] += 1
    
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
    
    # Can you just update this dynamically?
    # Well - you need to know all the edges whose status might have flipped.
    # At mixing this is likely very small compared to the total number of edges
    # but doesn't seem like a huge advantage.
            
    
    return contradictions



def step(graph, cut_edges, non_cut_edges, temperature, fugacity):

    coin = random.uniform(0,1)


    if coin < 1/2:
        return non_cut_edges, cut_edges

    current_contradictions = contradictions(graph, cut_edges)
    current_edge_cuts = len(cut_edges)
    current_num_colors = graph.graph["num_colors"]
    
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
        graph.graph["num_colors"] = current_num_colors
        return non_cut_edges, cut_edges
    else:
        return non_cut_edges, cut_edges

def is_connected_subgraph(graph, set_of_nodes):
    
    return 

def potts_step(graph, cut_edges, non_cut_edges, temperature, fugacity):
    
    proposed_node = random.choice(list(graph.nodes()))
    proposed_color = random.choice(list(graph.nodes()))
    old_color = graph.graph["coloring"][proposed_node]
    current_contradictions = contradictions(graph, cut_edges)
    current_edge_cuts = len(cut_edges)
    old_num_colors = graph.graph["num_colors"]
    
    # First, check that if there is a block of proposed color, it is adjacent
    # to proposed node
    
    color_block = set ( [ y for y in graph.nodes() if graph.graph["coloring"][y] == proposed_color])
    if color_block:
        adjacent = False
        proposed_node_neighbors = graph.neighbors(proposed_node)
        for z in proposed_node_neighbors:
            if z in color_block:
                adjacent = True
        if adjacent == False:
            # proposal was rejected
            return graph
    # Set color proposal
    
    graph.graph["coloring"][proposed_node] = proposed_color
    
    # Next, check that the old_color block is still connected
    color_block = set ( [ y for y in graph.nodes() if graph.graph["coloring"][y] == old_color])
    
    is_connected = is_connected_subgraph(graph, color_block)
    
    if not is_connected:
        # proposal was rejected
        graph.graph["coloring"][proposed_node] = old_color
    
    if is_connected:
        # proposal was accepted. Now we move to the Metropolis-Hasting's steps
    
        # first, update the edge cut sets:
            
            
            
        # Update the  number of colors used
        
        new_num_colors = graph.graph["num_colors"]
            
        # calculate the parameters and do MH:
        
        new_contradictions = contradictions(graph, cut_edges)
        new_edge_cuts = len(cut_edges)
    
        if new_contradictions <= current_contradictions:
            return non_cut_edges, cut_edges
    
        coin = random.uniform(0,1)
    
        # Todo: Calculate coloring ratio
        N = len(graph.nodes())
        
        coloring_ratio = math.comb(N, old_num_colors) / math.com(N, new_num_colors)
        # Get ratio right
        
        #
    
        if (temperature**(new_contradictions - current_contradictions))* (fugacity**(new_edge_cuts - current_edge_cuts))*coloring_ratio <= coin:    
            
            # Reset changes here:
                
                
                
            
            return non_cut_edges, cut_edges
        else:
            return non_cut_edges, cut_edges
        
def run_markov_chain(graph, steps = 100, temperature = .5, fugacity = 1):
    #Return a sample as [Cutedges, non_cutedges, coloring]

    #temperature = .8
    #steps = 1000
    graph.graph["ordered_edges"] = list(graph.edges())
    graph.graph["coloring"] = {x : x for x in graph.nodes()}
    graph.graph["num_colors"] = len(graph.nodes())
    
    cut_edges = set( graph.graph["ordered_edges"])
    non_cut_edges = set([])
    
    
    # cut_edges = set ([x for x in edges if np.random.binomial(1,.5) == 1])
    # non_cut_edges = set ( [ x for x in edges if x not in cut_edges])
    # If you do this you need to update the coloring
    
    for i in range(10 * len(graph.edges())):
        non_cut_edges, cut_edges = step(graph, cut_edges, non_cut_edges, 1, 1)
        # A burn in. We should replace this with a step that produces a random 
        # sample and updates the coloroing.
        # Do this to avoid over counting the number of successes from earlier
        # parts of the run.

    sample = False
    num_found = 0
    
    contradictions_histogram = {x : 0 for x in range(len(graph.edges()) + 1)}
    
    for i in range(steps):
        non_cut_edges, cut_edges = step(graph, cut_edges, non_cut_edges, temperature, fugacity)
        num_contradictions = contradictions(graph, cut_edges) 
        contradictions_histogram[num_contradictions] += 1
        if num_contradictions == 0:
            num_found += 1
            sample = [copy.deepcopy(cut_edges),copy.deepcopy(non_cut_edges), copy.deepcopy(graph.graph["coloring"])]
            #print('found one')
    plt.bar(list(contradictions_histogram.keys()), contradictions_histogram.values())
    plt.show()
    print(contradictions_histogram)
    print("found this many samples:" , num_found)

    return sample


def test_grid_graph():

    #for fugacity in [.2,.4,.5,.6,.8,1,2,3,4,5]:
    for fugacity in [1]:
        size = 3
        graph = nx.grid_graph([size, size])
        MC_steps = 100
        MC_temperature = .4
        print("Running with numsteps:", MC_steps ," and temperature: " , MC_temperature)
        print(str(fugacity))
        sample = run_markov_chain(graph, MC_steps, MC_temperature, fugacity)
        name = "markov_chain_with_fugacity" + str(fugacity)
        print(name)
        if sample != False:
            viz(graph, sample[1], sample[2], name)
        else:
            print("no sample found")
        print(graph.graph["num_colors"])
test_grid_graph()

#started 7:11

#Under 4 minutes.