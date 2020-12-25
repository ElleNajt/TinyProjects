# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 19:38:04 2020

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
import numba as nb
from itertools import compress

@nb.njit
def test_consistency(incidence_matrix,edges_to_remove):
    """
    Set J = 1 - edges_to_remove.
    
    This tests whether the submatrix of M := incidence_matrix defined by J, 
    M_J := incidence_matrix_copy[J], is a 'flat' of M, 
    i.e. whether every column of incidence_matrixin the column span of M_J
    is already included in M_J.

    Parameters
    ----------
    incidence_matrix : numpy array
        The incidence matrix of the graph
    edges_to_remove : array of booleans
        False when the edge is in the proposed flat, J.

    Returns
    -------
    bool
        whether or not 1 - J is a flat.

    """

    incidence_matrix_copy = np.copy(incidence_matrix)
    incidence_matrix_copy[edges_to_remove] = 0
    minor_rank = np.linalg.matrix_rank(incidence_matrix_copy)

    
    col = 0
    for i in edges_to_remove:
        # This iterates over the edges, and on each edge not in J
        # checks whether unzeroing the column corresponding to that edge
        # increases the rank.
        if i == True:
            incidence_matrix_copy[col] = incidence_matrix[col]
            new_rank = np.linalg.matrix_rank(incidence_matrix_copy)
            if new_rank == minor_rank:
                return False
            incidence_matrix_copy[col] = 0
        col += 1
    return True


@nb.njit
def rejection_sample(incidence_matrix, num_edges):
    edges_to_remove = np.random.choice(a=np.array([False, True]), size=num_edges)
    return 1 - edges_to_remove, test_consistency(incidence_matrix,edges_to_remove)

    
@nb.njit()
def test_rejection_sample(edge_list,num_edges):
    while True:
        new, success = rejection_sample(edge_list, num_edges)
        if success:
            return new



'''
Code below is only used for visualizations. It does not need optimization.
'''

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

def viz(graph, edge_set, coloring, name):


    convert = {}
    coloring_convert = {}
    node_list = list(graph.nodes())
    for i in range(len(node_list)):
        convert[node_list[i]] = i
    for i in range(len(node_list)):
        coloring_convert[node_list[i]] = convert[coloring[node_list[i]]]


    for x in graph.nodes():
        graph.nodes[x]["pos"] = (x[0],x[1])
    values = [1 - int(x in edge_set or (x[1], x[0]) in edge_set) for x in graph.edges()]
    node_values = [convert[coloring[x]] for x in graph.nodes()]
    f = plt.figure()
    print("drawing now")
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_color = 'w', edge_color = values, labels = coloring_convert, width = 4, node_size= 65, font_size = 10)
    print("saving now")
    f.savefig(name + "at_time_" + str(int(time.time())) + ".png")

    return

def convert_and_viz(edge_list, sample, graph):
    
    on_edges = list(compress(edge_list, [bool(x) for x in sample]))
    
    graph.graph["coloring"] = {x : x for x in graph.nodes()}

    for e in on_edges:
        graph = update_coloring(graph, e)

    coloring = graph.graph["coloring"]
    
    viz(graph, on_edges, coloring, "testing")

'''
End of visualization code
'''
    
samples_list = []
for i in range(1):
    current_time = time.time()
    graph = nx.grid_graph([4,4])
    incidence_matrix = nx.incidence_matrix(graph).A.T
    edge_list = list(graph.edges())
    num_edges = len(graph.edges())
    sample = test_rejection_sample(incidence_matrix,num_edges)
    samples_list.append(sample)
    print(time.time() - current_time)
    
convert_and_viz(edge_list, sample, graph)