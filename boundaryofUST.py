# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:36:10 2020

@author: lnajt
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 11:57:08 2018

@author: MGGG

"""

import networkx as nx
import random
import numpy as np
##############
    
'''Wilsons Algorithm'''

def random_spanning_tree_wilson(graph):
    '''The David Wilson random spanning tree algorithm'''
    tree_edges = []
    root = random.choice(list(graph.nodes()))
    hitting_set = set ( [ root])
    allowable_set = set(graph.nodes()).difference(hitting_set)
    len_graph = len(graph)
    len_hitting_set = 1
    while len_hitting_set < len_graph:
        start_node = random.choice(list(allowable_set))
        trip = random_walk_until_hit(graph, start_node, hitting_set)
        new_branch, branch_length = loop_erasure(trip)
        for i in range(branch_length - 1):
            tree_edges.append( [ new_branch[i], new_branch[i + 1]])
        for v in new_branch[:-1]:
            hitting_set.add(v)
            len_hitting_set += 1
            allowable_set.remove(v)
    tree = nx.DiGraph()

    for node in graph.nodes:
        node_attributes = list(graph.nodes[node].keys())
        tree.add_node(node)
        for attr in node_attributes:
            tree.nodes[node][attr] = graph.nodes[node][attr]
    tree.add_edges_from(tree_edges)
    return tree


def simple_random_walk_variable_length(graph,node, walk_length):
    '''does a random walk of length walk_length'''
    wet = set([node])
    trip = [node]
    while len(wet) < walk_length:
        next_step = random.choice(list(graph.neighbors(node)))
        wet.add(next_step)
        trip.append(next_step)
        node = next_step
    return trip, wet

def forward_tree_variable_length(graph,node, walk_length):
    '''builds the forward tree in Broders algorithm, using a walk of length
    walk_length'''
    
    walk, wet = simple_random_walk_variable_length(graph, node, walk_length)
    edges = []
    for vertex in list(wet):
        if (vertex != walk[0]):
            first_occurance = walk.index(vertex)
            edges.append( [walk[first_occurance], walk[first_occurance-1]])
    return edges, wet

def random_tree_variable_length(graph, walk_length):
    '''runs Broders algorithm to produce a tree of length walk_length'''
    tree_edges, wet = forward_tree_variable_length(graph, random.choice(list(graph.nodes())), walk_length)
    tree = nx.DiGraph()
    for node in list(wet):
        tree.add_node(node) 

    tree.add_edges_from(tree_edges)
    return tree


def random_spanning_tree_wilson_with_starting(graph, starting_tree):
    #The David Wilson random spanning tree algorithm
    tree_edges = list(starting_tree.edges())
    hitting_set = set(starting_tree.nodes())
    allowable_set = set(graph.nodes()).difference(hitting_set)
    len_graph = len(graph)
    len_hitting_set = len(hitting_set)
    while len_hitting_set < len_graph:
        start_node = random.choice(list(allowable_set))
        trip = random_walk_until_hit(graph, start_node, hitting_set)
        new_branch, branch_length = loop_erasure(trip)
        #print(branch_length)
        for i in range(branch_length - 1):
            tree_edges.append( [ new_branch[i], new_branch[i + 1]])
        for v in new_branch[:-1]:
            hitting_set.add(v)
            len_hitting_set += 1
            allowable_set.remove(v)
    tree = nx.DiGraph()

    tree.add_edges_from(tree_edges)
    return tree

def random_walk_until_hit(graph, start_node, hitting_set):
    '''Does a random walk from start_node until it hits the hitting_set
    
    :graph: input graph
    :start_node: the node taht the graph starts at
    :hitting_set: the set to stop at, i.e. the tree we are building up
    
    '''
    
    current_node = start_node
    trip = [current_node]
    while current_node not in hitting_set:
        current_node = random.choice(list(graph.neighbors(current_node)))
        trip.append(current_node)
    return trip

def loop_erasure(trip):
    '''erases loops from a trip
    
    :trip: input of node names...
    '''
    n = len(trip)
    loop_erased_walk_indices = []
    last_occurance = n - trip[::-1].index(trip[0]) - 1
    loop_erased_walk_indices.append(last_occurance)
    branch_length = 0
    while trip[loop_erased_walk_indices[-1]] != trip[-1]:
        last_occurance = n -  trip[::-1].index(trip[loop_erased_walk_indices[-1]])  -1
        loop_erased_walk_indices.append(last_occurance + 1)
        branch_length += 1
    loop_erased_trip = [trip[i] for i in loop_erased_walk_indices]
    
    return (loop_erased_trip, branch_length + 1)

def statistics():
    samples = 100
    graph = nx.grid_graph([20,20])
    W_trees = []
    for i in range(samples):
        W_trees.append( nx.to_undirected(random_spanning_tree_wilson(graph)))
        
    