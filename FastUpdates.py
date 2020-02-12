# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:36:58 2019

@author: Lorenzo
"""

import networkx as nx
import Facefinder
import random
import numpy as np
import matplotlib.pyplot as plt
def face_edges(face):
    '''
    Faces are presently stored as an oriented cycle of nodes. 
    This outputs a cycle of edges.
    
    '''
    edge_list = []
    for i in range(len(face)):
        edge_list.append(frozenset( {face[i], face[(i + 1) % len(face)]}  )  )
    return edge_list

def proposal(graph,face,cycle):
    '''
    face::A set of edges that we propose the flips for
    cycle:: The current set of edges in the simple cycle, stored as a function
    on edges(graph).
    '''
    common_edges = []
    uncommon_edges = []
    cut_edges = face_edges(face)
    common_edges_format = []
    for e in cut_edges:
        if cycle[e] == 1:
            common_edges.append(e)
            common_edges_format.append((list(e)[0], list(e)[1]))
        else:
            uncommon_edges.append(e)
    common_nodes = []
    uncommon_nodes = []
    for e in cut_edges:
        list_e = list(e)
        for i in [0,1]:
            v = list_e[i]
            if cycle[v] == 1:
                common_nodes.append(v)
            else:
                uncommon_nodes.append(v)
    common_nodes = list(set(common_nodes))
            
    face_graph = graph.edge_subgraph(common_edges_format)
    
    for v in face_graph.nodes():
        #print(v)
        if v in common_nodes:
            common_nodes.remove(v)
    
    if len(common_nodes) > 0:
        return [cycle, False]
    
    if len(common_edges) == 0:
        return [cycle, False]
    if len(common_edges) == len(cut_edges):
        #print(common_edges, cut_edges)
        return [cycle, False]
    
    if nx.is_connected(face_graph):
        
        
        ###Update edges:
            
        for e in cut_edges:
            if cycle[e] == 0:
                cycle[e] = 1
            else:
                cycle[e] = 0
        
        ###Update nodes:
        common_nodes = []
        uncommon_nodes = []
        for e in cut_edges:
            list_e = list(e)
            for i in [0,1]:
                v = list_e[i]
                if cycle[v] == 1:
                    common_nodes.append(v)
                else:
                    uncommon_nodes.append(v)
        for v in uncommon_nodes:
            cycle[v] = 1
        for v in common_nodes:
            cycle[v] = 0
        
        degree_one_nodes = [x for x in face_graph.nodes() if face_graph.degree(x) == 1]
        for v in degree_one_nodes:
            cycle[v] = 1
        return [cycle, True]
    else:
        return [cycle, False]
    
    
def preprocess(graph):
    graph = Facefinder.compute_rotation_system(graph)
    graph = Facefinder.compute_all_faces(graph)
    return graph


def run_steps(graph, cycle, steps = 10):
    successes = 0
    for step in range(steps):
        face = random.choice(list(graph.graph["faces"]))
        cycle, success = proposal(graph, face, cycle)
        #print(success, face)
        successes += success
    print(successes)
    return cycle

def run_chain():
    m = 40
    steps = 10000
    print("starting preprocessing")
    graph = nx.grid_graph([m,m])
    graph.name = "grid_size:" + str(m)
    for x in graph.nodes():
        
        graph.node[x]["pos"] = np.array([x[0], x[1]])
    graph = preprocess(graph)
    cycle = dict()
    for e in graph.edges():
        cycle[frozenset(e)] = 0
    cycle[frozenset( [(0,0), (1,0)] )] = 1
    cycle[frozenset( [(0,0), (0,1)])] = 1
    cycle[frozenset( [(1,1), (1,0) ])] = 1
    cycle[frozenset( [(1,1), (0,1)])] = 1  
    for v in graph.nodes():
        cycle[v] = 0
    for e in graph.edges():
        if cycle[frozenset(e)] == 1:
            cycle[e[0]] = 1
            cycle[e[1]] = 1
    print("preprocessing done")
    cycle = run_steps(graph, cycle, steps)
    viz(graph, cycle)
    return cycle

def subgraph_is_simpel_cycle(graph, cycle):
    '''
    Doublechecks that the induced subgraph is a cycle
    '''
    edge_set = []
    for e in graph.edges():
        if cycle[frozenset(e)] == 1:
            edge_set.append(e)
            
    subgraph = graph.edge_subgraph(edge_set)
    if nx.is_connected(subgraph) and nx.degree_histogram(subgraph)[2] == len(subgraph):
        print("Is a cycle")

def viz(graph, cycle):
    k = 20

    values = [int(cycle[frozenset(x)] == 1 ) for x in graph.edges()]

    nx.draw(graph, pos=nx.get_node_attributes(graph, "pos"), node_size = 1, width =2, cmap=plt.get_cmap('magma'),  edge_color=values)

    
cycle = run_chain()