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
import scipy as sc
import numba as nb
from numba.typed import Dict
from numba.typed import List

params_default = nb.typed.Dict.empty(
    key_type=nb.typeof('1'),
    value_type=nb.typeof('1')
)

@nb.njit
def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

@nb.njit #(nb.typed.Dict(), nb.typeof(int(1)))
def update_coloring(graph_nodes, coloring, num_colors, a, b):
    #Here we add edge to the set of non-cut edges, meaning that the two endpoints are now definately in the same connected compoennt. We update the coloring to reflect that.
    # print("here")
    # a = coloring[edge[0]]
    # b = coloring[edge[1]]

    #if a == b:
    #    return coloring, num_colors
    #    #If already in the same component, do nothing.
    #else:
    for x in graph_nodes:
        print(x, a, b)
        if coloring[x] == b:
            coloring[x] = a
    num_colors += -1
    return coloring, num_colors
    

@nb.njit
def component(component_superset, node, edge_subset):
    # Return the set of nodes of graph that are in the same connected component
    # as node , using edges edge_subset. Given color_component as extra information
    # which is a set of nodes containing the component.
    # print(graph_edges, graph_nodes,  neighbors, component_superset, node, edge_subset)
    # print(component_superset, node, edge_subset)
    known_component =set( [node])
    frontier = set([node])
    component_superset.remove(node)
    
    while frontier:

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
    return known_component

@nb.njit
def coloring_split(graph_edges, graph_nodes, coloring, num_colors, non_cut_edges, edge):

    #This removes edge from the coloring -- if that was the bridge between the component of that coloring, it assigns an unused color to one of the components.

    a = coloring[edge[0]]
    b = coloring[edge[1]]
    #We know a == b...
    if a != b:
        print("something went wrong!")
        
    color_component = set()
    for x in graph_nodes:
        if coloring[x] == a:
            color_component.add(x)

    color_subgraph_edge_list = List()
    is_empty = True
    for f in non_cut_edges:
        # Note that edge is not in this set
        if coloring[f[1]] == a:
            is_empty = False
            color_subgraph_edge_list.append(f)
    
    if not is_empty:
        b_component = component(color_component, edge[1], color_subgraph_edge_list)
    else:
        b_component = set( [edge[1]])

    if edge[0] in b_component:
        #This was the case that e was not a bridge
        return coloring, num_colors

    # Now, in the case that e was a bridge, we need to reassign the colors. 
    # First we find a color not used by the other components.

    possible_colors = set(graph_nodes)
    possible_colors = remove_values(coloring, possible_colors)
    print(possible_colors)
    #print(coloring)
    #for used_color in coloring.values():
    #    print(possible_colors)
    #    if used_color in possible_colors:
    #        possible_colors.remove(used_color)
    #print(possible_colors)
    unused_color = possible_colors.pop()
    #We let component with edge[0] keep its color, and reassign compoennt with edge[1]
    for y in b_component:
        coloring[y] = unused_color
    num_colors += 1
    
    return coloring, num_colors

@nb.njit
def remove_values(dictionary, input_set):
    #removes values of the dictionary from the input set
    print(dictionary, input_set)
    for value in dictionary.values():
        print(input_set)
        print(value)
        if value in input_set:
            input_set.remove(value)
    print(input_set)
    return input_set
    

@nb.njit
def calculate_contradictions(coloring, cut_edges):
    
    contradictions = 0
    for edge in cut_edges:
        if coloring[edge[0]] == coloring[edge[1]]:
            contradictions += 1
    
    return contradictions

def non_jitted_step(graph_edges, graph_nodes,  neighbors, coloring, num_colors, contradictions,  cut_edges, non_cut_edges, MC_temperature, fugacity):

    current_contradictions = contradictions
    current_edge_cuts = len(cut_edges)
    # original_coloring = copy.copy(coloring)
    original_num_colors = num_colors
    
    edge_index = int(np.random.uniform(0, len(graph_edges)))
    edge = graph_edges[edge_index]

    if edge in non_cut_edges:
        add_edge = False
        non_cut_edges.remove(edge)
        cut_edges.add(edge)
        coloring , num_colors = coloring_split(graph_edges, graph_nodes, coloring, num_colors, non_cut_edges, edge)
    else:
        add_edge = True
        non_cut_edges.add(edge)
        cut_edges.remove(edge)
        coloring, num_colors = update_coloring(graph_nodes, coloring, num_colors, coloring[edge[0]], coloring[edge[1]])

    new_contradictions = calculate_contradictions(coloring, cut_edges)
    new_edge_cuts = len(cut_edges)

    coin = random.uniform(0,1)
    print((MC_temperature**(new_contradictions - current_contradictions))* (fugacity**(new_edge_cuts - current_edge_cuts)))
    if (MC_temperature**(new_contradictions - current_contradictions))* (fugacity**(new_edge_cuts - current_edge_cuts)) <= coin:        
        if add_edge == True:
            non_cut_edges.remove(edge)
            cut_edges.add(edge)
            graph = coloring_split(graph_edges, graph_nodes, coloring, num_colors, non_cut_edges, edge)
        if add_edge == False:
            non_cut_edges.add(edge)
            cut_edges.remove(edge)
            graph = update_coloring(graph_nodes, coloring, num_colors, coloring[edge[0]], coloring[edge[1]])
        # coloring = original_coloring
        num_colors = original_num_colors 
        return non_cut_edges, cut_edges, coloring, num_colors, contradictions
    else:
        contradictions = new_contradictions
        # if graph.graph["contradictions"] != calculate_contradictions(graph, cut_edges):
        #    print("misalignmnent occurred _ edgeC")
        return non_cut_edges, cut_edges, coloring, num_colors, contradictions



@nb.njit
def step(graph_edges, graph_nodes,  neighbors, coloring, num_colors, contradictions,  cut_edges, non_cut_edges, MC_temperature, fugacity, non_cut_edges_nonempty):

    current_contradictions = contradictions
    current_edge_cuts = len(cut_edges)
    # original_coloring = copy.copy(coloring)
    original_coloring = Dict()
    for x in coloring.keys():
        original_coloring[x] = coloring[x]
    
    original_num_colors = num_colors
    
    edge_index = int(np.random.uniform(0, len(graph_edges)))
    edge = graph_edges[edge_index]
    
    print(edge in non_cut_edges)

    if edge in non_cut_edges:
            add_edge = False
            non_cut_edges.remove(edge)
            cut_edges.add(edge)
            coloring , num_colors = coloring_split(graph_edges, graph_nodes, coloring, num_colors, non_cut_edges, edge)
    else:
        add_edge = True
        non_cut_edges.add(edge)
        cut_edges.remove(edge)
        coloring, num_colors = update_coloring(graph_nodes, coloring, num_colors, coloring[edge[0]], coloring[edge[1]])

    new_contradictions = calculate_contradictions(coloring, cut_edges)
    new_edge_cuts = len(cut_edges)
    # print(new_contradictions, current_contradictions)
    coin = np.random.uniform(0,1)
    # print((MC_temperature**(new_contradictions - current_contradictions))* (fugacity**(new_edge_cuts - current_edge_cuts)), coin, coin >= (MC_temperature**(new_contradictions - current_contradictions))* (fugacity**(new_edge_cuts - current_edge_cuts)))
    if (MC_temperature**(new_contradictions - current_contradictions))* (fugacity**(new_edge_cuts - current_edge_cuts)) <= coin:        
        if add_edge == True:
            non_cut_edges.remove(edge)
            cut_edges.add(edge)
            # graph = coloring_split(graph_edges, graph_nodes, coloring, num_colors, non_cut_edges, edge)
        if add_edge == False:
            non_cut_edges.add(edge)
            cut_edges.remove(edge)
            # graph = update_coloring(graph_nodes, coloring, num_colors, coloring[edge[0]], coloring[edge[1]])
        coloring = original_coloring
        num_colors = original_num_colors 
        return non_cut_edges, cut_edges, coloring, num_colors, contradictions
    else:
        # print("accepted", edge, non_cut_edges, cut_edges, add_edge)
        contradictions = new_contradictions
        # if graph.graph["contradictions"] != calculate_contradictions(graph, cut_edges):
        #    print("misalignmnent occurred _ edgeC")
        return non_cut_edges, cut_edges, coloring, num_colors, contradictions

def is_connected_subgraph(graph, set_of_nodes, node):
    
    size_of_set = len(set_of_nodes)
    known_component =set( [node])
    frontier = set([node])
    component_superset = set_of_nodes
    component_superset.remove(node)

    while frontier:

        new_frontier = set()
        remaining_nodes = set()
        for z in component_superset:
            connected = False
            for w in frontier:
                if z in graph.neighbors(w):
                    known_component.add(z)
                    new_frontier.add(z)
                    connected = True
            if connected == False:
                remaining_nodes.add(z)
        frontier = new_frontier
        component_superset = remaining_nodes
    return len(known_component) == size_of_set

def flip_node(graph, proposed_node, old_color, new_color, non_cut_edges, cut_edges):
    proposed_node_neighbors = graph.neighbors(proposed_node)    
    for z in proposed_node_neighbors:
        if graph.graph["coloring"][z] == old_color:
            if (proposed_node, z) in graph.graph["ordered_edges"]:
                # We want to keep the set of ordered edges the same,
                # so that the edge flip chain works correctly.
                cut_edges.add( (proposed_node,z) )
            elif (z, proposed_node) in graph.graph["ordered_edges"]:
                cut_edges.add( (z,proposed_node) )
            if (proposed_node, z) in non_cut_edges:
                non_cut_edges.remove( (proposed_node, z))
            if (z, proposed_node) in non_cut_edges:
                non_cut_edges.remove( (z, proposed_node))
        if graph.graph["coloring"][z] == new_color:
            if (proposed_node, z) in graph.graph["ordered_edges"]:
                non_cut_edges.add( (proposed_node,z) )
            elif (z, proposed_node) in graph.graph["ordered_edges"]:
                non_cut_edges.add( (z,proposed_node) )
            if (proposed_node, z) in cut_edges:
                cut_edges.remove( (proposed_node, z))
            if (z, proposed_node) in cut_edges:
                cut_edges.remove( (z, proposed_node))
    return non_cut_edges, cut_edges

def potts_step(graph, cut_edges, non_cut_edges, temperature, fugacity):
    '''
    Optimizations:
        
        Use the geometric skips technique:
            1. Calculate set of (node, color) pairs that pass the first
            connectivity test
            2. This should dynamically updatable?
            
    '''
    
    
    proposed_node = random.choice(list(graph.nodes()))
    proposed_color = random.choice(list(graph.nodes()))
    #print(proposed_node, proposed_color)
    old_color = graph.graph["coloring"][proposed_node]
    current_contradictions = graph.graph["contradictions"]
    # current_contradictions =  calculate_contradictions(graph, cut_edges)
    current_edge_cuts = len(cut_edges)
    old_num_colors = graph.graph["num_colors"]
    old_num_colors = len( set ( [graph.graph["coloring"][x] for x in graph.nodes()]))
                         
    # If there is a block of proposed_color, it must be adjacent to proposed_node
    
    color_block = set ( [ y for y in graph.nodes() if graph.graph["coloring"][y] == proposed_color])
    if color_block:
        proposed_node_neighbors = graph.neighbors(proposed_node)
        for z in proposed_node_neighbors:
            if z in color_block:
                break
        else:
            # proposal is not a connected partition, so reject.
            return non_cut_edges, cut_edges
        
    # Set color proposal
    
    graph.graph["coloring"][proposed_node] = proposed_color
    
    # Next, check that the old_color block is still connected
    color_block = [ y for y in graph.nodes() if graph.graph["coloring"][y] == old_color]
    
    if color_block != []:
        a_node = color_block[0]
        is_connected = is_connected_subgraph(graph, color_block, a_node)
        if not is_connected:
            # proposal is not connected, so reject 
            graph.graph["coloring"][proposed_node] = old_color
            return non_cut_edges, cut_edges

    # proposal was accepted. Thus far it would give a uniform connected coloring.
    
    
    # Now we move to the Metropolis-Hasting's steps

    # first, update the edge cut sets:
    non_cut_edges, cut_edges = flip_node(graph, proposed_node, old_color, proposed_color, non_cut_edges, cut_edges)
        
    # Update the  number of colors used
    
    # new_num_colors = graph.graph["num_colors"]
    new_num_colors = len( set ( graph.graph["coloring"].values()))
    
    # calculate the parameters and do MH:
    
    new_contradictions = graph.graph["contradictions"] # calculate_contradictions(graph, cut_edges)
    # This may be constant ? 
    
    new_edge_cuts = len(cut_edges)

    # Calculate ratio that reweights according to the coloring options
    # We want to weight by f(P) = math.comb(N, num_colors)^{-1}
    
    N = len(graph.nodes())
    old_energy = binom(N, old_num_colors)**(-1)
    new_energy = binom(N, new_num_colors)**(-1)
    coloring_ratio = min(1,  new_energy / old_energy) # old_energy / new_energy #
 
    #
    coin = random.uniform(0,1)
    # print(coloring_ratio)
    #if (temperature**(new_contradictions - current_contradictions))* (fugacity**(new_edge_cuts - current_edge_cuts))*coloring_ratio >= coin:    
    if coin >= coloring_ratio:    
        # Rejection.
        
        #Now reset the changes:
        graph.graph["coloring"][proposed_node] = old_color     
        non_cut_edges, cut_edges  = flip_node(graph, proposed_node, proposed_color, old_color, non_cut_edges, cut_edges)
        return non_cut_edges, cut_edges
    else:
        # acceptance
        graph.graph["contradictions"] = new_contradictions
        return non_cut_edges, cut_edges

def parallel_tempering(graph, steps, temperatures, temperature_weights, fugacities, fugacities_weights):
    graph.graph["ordered_edges"] = list(graph.edges())
    graph.graph["coloring"] = {x : x for x in graph.nodes()}
    graph.graph["num_colors"] = len(graph.nodes())
    cut_edges = set( graph.graph["ordered_edges"])
    non_cut_edges = set([])
    graph.graph["contradictions"] = 0    
    for i in range(10 * len(graph.edges()) * int(math.log( len(graph.edges())))):
        non_cut_edges, cut_edges = step(graph, cut_edges, non_cut_edges, 1, 1)
    
    
    sample = False
    num_found = 0
    contradictions_histogram = {x : 0 for x in range(len(graph.edges()) + 1)}
    
    current_temperature = temperatures[0]
    current_fugacity = fugacities[0]
    
    for i in range(steps):
        
        coin = random.random()
        if coin < .5:
            non_cut_edges, cut_edges = step(graph, cut_edges, non_cut_edges, temperature, fugacity)
            non_cut_edges, cut_edges = potts_step(graph, cut_edges, non_cut_edges, temperature, fugacity)
        else:
            # pick a new temperature and fugacy ... ?
            return
        num_contradictions = graph.graph["contradictions"] 
        contradictions_histogram[num_contradictions] += 1
        if num_contradictions == 0:
            num_found += 1
            sample = [copy.copy(cut_edges),copy.copy(non_cut_edges), copy.copy(graph.graph["coloring"])]

    #plt.bar(list(contradictions_histogram.keys()), contradictions_histogram.values())
    #plt.show()
    print("found this many samples:" , num_found)

# @nb.njit
def viz(graph, edge_set, coloring, name, location_mapping):


    convert = {}
    coloring_convert = {}
    node_list = list(graph.nodes())
    for i in range(len(node_list)):
        convert[node_list[i]] = i
    for i in range(len(node_list)):
        coloring_convert[node_list[i]] = convert[coloring[node_list[i]]]


    for x in graph.nodes():
        graph.nodes[x]["pos"] = location_mapping[x]
    values = [1 - int(x in edge_set or (x[1], x[0]) in edge_set) for x in graph.edges()]
    node_values = [convert[coloring[x]] for x in graph.nodes()]
    f = plt.figure()
    print("drawing now")
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_color = 'w', edge_color = values, labels = coloring_convert, width = 4, node_size= 65, font_size = 10)
    print("saving now")
    f.savefig(name + "at_time_" + str(int(time.time())) + ".png")

    return 

#@nb.njit
def run_markov_chain(graph, graph_edges, graph_nodes,  neighbors, MC_steps = 100, MC_temperature = .5, fugacity = 1):
    '''
    Optimizations todo:
        move everything we call out of networkx objects
        e.g. have networkx prepare the adjacency and incidence graphs and incidence dictionary
        then, move the corresponding to calculations to cython optimized code.
    
    '''

    
    # temp = {0: 0, 1: 1, 2: 2, 3: 0, 4: 0, 5: 2, 6: 6, 7: 7, 8: 8}
    # for x in temp.keys():
    #    coloring[x] = temp[x]

    coloring = Dict() 
    for x in graph_nodes:
        coloring[x] = x
    cut_edges = set(graph_edges)
    non_cut_edges = set([])
    contradictions = 0
    
    sample = False
    num_found = 0
    num_colors= len(graph_nodes)
    contradictions_histogram = Dict()
    for x in range(len(graph_edges) + 1):
        contradictions_histogram[x] = 0
        
    
    non_cut_edges, cut_edges, coloring, num_colors, contradictions = non_jitted_step(graph_edges, graph_nodes,  neighbors, coloring, num_colors, contradictions, cut_edges, non_cut_edges, MC_temperature, fugacity)
    
    for i in range(MC_steps):
        # print("step", i)
        # print(bool(non_cut_edges))
        #if bool(non_cut_edges):
        #    non_cut_edges, cut_edges, coloring, num_colors, contradictions = step(graph_edges, graph_nodes,  neighbors, coloring, num_colors, contradictions, cut_edges, non_cut_edges, MC_temperature, fugacity, bool(non_cut_edges))
        #else:
        
        non_cut_edges, cut_edges, coloring, num_colors, contradictions = non_jitted_step(graph_edges, graph_nodes,  neighbors, coloring, num_colors, contradictions, cut_edges, non_cut_edges, MC_temperature, fugacity)
            
        # print(non_cut_edges)
        
        # viz(graph, non_cut_edges, coloring, name, location_mapping)
        # print(coloring)
        contradictions_histogram[contradictions] += 1
        if contradictions == 0:
            num_found += 1
            sample = [copy.copy(cut_edges),copy.copy(non_cut_edges),copy.copy(coloring)]
    #print(contradictions_histogram)
    print("found this many samples:" , num_found)

    return sample


def test_grid_graph(size = 10, MC_steps = 10000, MC_temperature = .8, fugacities = [1]):
    print("reminder -- need to understand the potts step on the contradictory ones")

    for fugacity in fugacities:
        print("Running on n = ", str(size), " with numsteps:", MC_steps ," and temperature: " , MC_temperature, "and fugacity: ", str(fugacity))
        
        
        graph = nx.grid_graph([size, size])
        
        graph_nodes = list(graph.nodes())
        mapping_inv = {}
        mapping = {}
        for i in range(len(graph_nodes)):
            mapping_inv[i] = graph_nodes[i]
            mapping[graph_nodes[i]] = i
        
        location_mapping = {mapping[x] : (x[0], x[1]) for x in graph.nodes()}

        graph = nx.relabel_nodes(graph, mapping)
        
        
        graph_edges = list(graph.edges())
        graph_nodes = list(graph.nodes())
        
        neighbors = Dict.empty(key_type=nb.typeof(1), value_type=nb.typeof( np.asarray([ 1,2] )))
        
        
        
        for x in graph_nodes:
            nbr_list = []
            for y in graph.neighbors(x):
                nbr_list.append(y)
            neighbors[x] = np.asarray(nbr_list)
            
        name = "markov_chain_ on n = " + str(size) + "with_fugacity" + str(fugacity) + "temp_" + str(MC_temperature) + "_steps_" +str(MC_steps) +"___" 
        
        sample = run_markov_chain(graph, graph_edges, graph_nodes,  neighbors, MC_steps, MC_temperature, fugacity)

        #print(name)
        if sample != False:
            print("done")
            viz(graph, sample[1], sample[2], name, location_mapping)
        else:
            print("no sample found")
            
#if __name__ == '__main__':
            
size = 3
MC_steps = 100
MC_temperature = .8
fugacity = 1


test_grid_graph(10, 100, 1, [1])

#test_grid_graph(10, 10000, .6, [1], burn_in = True, start_at_min_part = False)