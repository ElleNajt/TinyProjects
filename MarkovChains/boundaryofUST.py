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
import copy
import matplotlib.pyplot as plt
import Facefinder
import os

#os.cwd('./Documents/GitHub/TinyProjects/MarkovChains')

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

def simple_cycle(unicycle):
    #returns the simple cycle of a unicycle
    #cycles= nx.simple_cycles(unicycle)
    bridges = list(nx.bridges(unicycle))

    edge_list = list(unicycle.edges())
    cycle_edges = [e for e in edge_list if e not in bridges]
    return cycle_edges

def find_supernode(graph):
    degree_list = [ nx.degree(graph)[v] for v in graph.nodes()]
    max_degree = max(degree_list)
    for v in graph.nodes():
        if nx.degree(graph)[v] == max_degree:
            return v
    return False

def statistics():
    trials = 5000
    samples = trials*200
    m= 5
    graph = nx.grid_graph([m,m])
    graph.name = "grid_size:" + str(m)
    for x in graph.nodes():
        graph.nodes[x]["pos"] = np.array([x[0], x[1]])



    dual = Facefinder.planar_dual(graph)
    W_trees = []
    branches = []
    for i in range(samples):
        W_trees.append( nx.to_undirected(random_spanning_tree_wilson(dual)))

    supernode = find_supernode(dual)
    boundary_faces = list(dual.neighbors(supernode))
    face_1 = boundary_faces[0]
    face_2 = boundary_faces[ int(len(boundary_faces)/2) + 1]

    if (face_1, supernode) or (supernode, face_1) in dual.edges():
        print("OK")
    if (face_2, supernode) or (supernode, face_2) in dual.edges():
        print("OK2")

    cycles = []
    for tree in W_trees:
        available_edges = set(dual.edges())
        available_edges = available_edges - set(tree.edges())
        e = random.choice(list(available_edges))
        unicycle = copy.deepcopy(tree)
        unicycle = nx.Graph(unicycle)
        unicycle.add_edge( e[0], e[1])
        cycles.append(simple_cycle(unicycle))

    cycles_containing_prescribed_faces = []
    corresponding_walks = []

    boundary_faces_frequencies = {}
    for face_a in boundary_faces:
        for face_b in boundary_faces:
            boundary_faces_frequencies[ (face_a, face_b)] = 0

    print("testing", boundary_faces_frequencies[ ( face_1, face_2)])

    for cycle in cycles:
        #faces = [x[0] for x in cycle] + [x[1] for x in cycle]
        #faces = set(faces)
        #print(faces)
        for face_a in boundary_faces:
            for face_b in boundary_faces:
                if (face_a, supernode) in cycle or (supernode, face_a) in cycle:
                    if (face_b, supernode) in cycle or (supernode, face_b) in cycle:
                        boundary_faces_frequencies[(face_a, face_b)] +=1

        face_a = face_1
        face_b = face_2
        if (face_a, supernode) in cycle or (supernode, face_a) in cycle:
            if (face_b, supernode) in cycle or (supernode, face_b) in cycle:
                #print('1')
                cycles_containing_prescribed_faces.append(cycle)
                walk = nx.Graph(nx.edge_subgraph(dual, cycle))
                walk.remove_node(supernode)
                corresponding_walks.append(walk)

    #print("testing2", boundary_faces_frequencies[ (face_1, face_2)])
    #print(corresponding_walks[0].edges())
    #print(len(cycles_containing_prescribed_faces))

    #print(boundary_faces_frequencies)

    LERW = []
    dual.remove_node(supernode)
    #Because we are testing whether the distributions looks like a LERW in the grid
    #portion of the dual graph
    for i in range(trials):
        trip = random_walk_until_hit(dual, face_1, set([face_2]))
        new_branch, branch_length = loop_erasure(trip)
        LERW.append(new_branch)

    return LERW, corresponding_walks, dual, boundary_faces_frequencies


def convert_node_seq_to_edge(graph, path):

    graph_edges = list(graph.edges())

    path_edges = []
    for i in range(len(path) -1 ):
        e = (path[i], path[i+1])
        if e in graph_edges:
            path_edges.append( (path[i], path[i+1]))
        else:
            e_flip = (e[1], e[0])
            if e_flip in graph_edges:
                path_edges.append( e_flip)
            else:

                print("somethings wrong:", e, e_flip)
    return path_edges

def viz(graph, path_edges):
    k = 20

    graph_edges = list(graph.edges())

    values = [((x in path_edges) or ( (x[1], x[0]) in path_edges)) for x in graph_edges]

    nx.draw(graph, pos=nx.get_node_attributes(graph, "pos"), node_size = 1, width =2, cmap=plt.get_cmap('magma'),  edge_color=values)

LERW, walks, dual, freq = statistics()

print("num_walks",len(walks))
print("num_LERW", len(LERW))
for test in [LERW, walks]:
    print(np.mean( [ len(x) for x in test]))

graph = dual
#for i in range(10):
#    viz(dual, convert_node_seq_to_edge(graph,LERW[i]))
#    viz(dual, list(walks[i].edges()))
