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


def coloring_split(graph, non_cut_edges, edge):

    #This removes edge from the coloring -- if that was the bridge between the component of that coloring, it assigns an unused color to one of the components.

    coloring = graph.graph["coloring"]
    a = coloring[edge[0]]
    b = coloring[edge[1]]
    #We know a == b...
    if a != b:
        print("something went wrong!")
        print(non_cut_edges)
        print(edge)
        print(coloring)
        viz(graph, non_cut_edges, graph.graph["coloring"])
        return None

    color_component = [x for x in graph.nodes() if coloring[x] == a]

    color_subgraph = nx.Graph(nx.subgraph(graph, color_component))
    #Passing to nx.Graph is necessary to make a mutable subgraph

    color_subgraph_edge_list = list(color_subgraph.edges())
    for f in color_subgraph_edge_list:
        #We need this loop so that we don't accidentally throw in all of the induced edges of the color component
        if f not in non_cut_edges:
            if (f[1], f[0]) not in non_cut_edges:
                color_subgraph.remove_edge(f[0], f[1])

    #color_subgraph.remove_edge(edge[0], edge[1])

    components = list(nx.connected_components(color_subgraph))

    if len(components) == 1:
        #This was the case that e was not a bridge
        return graph

    #Now, in the case that e was a bridge, we need to reassign the colors. First we find a color not used by the other components.

    possible_colors = set(graph.nodes())
    for used_color in coloring.values():
        if used_color in possible_colors:
            possible_colors.remove(used_color)

    unused_color = list(possible_colors)[0]


    #We let component[0] keep its color, and reassign the colors in component[1].
    for y in components[1]:
        #print(y)
        coloring[y] = unused_color

    graph.graph["coloring"] = coloring
    return graph


def contradictions(graph, cut_edges):

    coloring = graph.graph["coloring"]
    contradictions = 0
    for e in cut_edges:
        if coloring[e[0]] == coloring[e[1]]:
            contradictions += 1
    return contradictions

def step(graph, cut_edges, non_cut_edges, temperature):

    coin = random.uniform(0,1)


    if coin < 1/2:
        return non_cut_edges, cut_edges

    old_non_cut_edges = copy.deepcopy(non_cut_edges)
    old_cut_edges = copy.deepcopy(cut_edges)
    old_coloring = copy.deepcopy(graph.graph["coloring"])

    current_contradictions = contradictions(graph, cut_edges)

    e = random.choice(graph.graph["ordered_edges"])

    if e in non_cut_edges:
        #print(e, " is now in cut set!")
        non_cut_edges.remove(e)
        cut_edges.add(e)
        graph = coloring_split(graph, non_cut_edges, e)
        #This is the update coloring that potentially reassigns the colors, because when making e cut some new components might emerge...
        #Be sure to use if else here, otherwise it just undoes itself :-D
    else:
        #print(e, "is now in merge set!")
        non_cut_edges.add(e)
        cut_edges.remove(e)
        graph = update_coloring(graph, e)
        #This is the update coloring that merges the two colors, based on moving e into non-cut.

    #viz(graph, non_cut_edges, graph.graph["coloring"])

    new_contradictions = contradictions(graph, cut_edges)
    #print(new_contradictions)

    if new_contradictions <= current_contradictions:
        return non_cut_edges, cut_edges

    coin = random.uniform(0,1)

    if temperature**(new_contradictions - current_contradictions) <= coin:
        #print("reset, at threshold:", temperature**(new_contradictions - current_contradictions) )
        non_cut_edges = old_non_cut_edges
        cut_edges = old_cut_edges
        graph.graph["coloring"] = old_coloring
        return non_cut_edges, cut_edges
    else:
        return non_cut_edges, cut_edges


def run_markov_chain(graph, steps = 100, temperature = .5):
    #Return a sample as [Cutedges, non_cutedges, coloring]

    #temperature = .8
    #steps = 1000
    graph.graph["ordered_edges"] = list(graph.edges())
    graph.graph["coloring"] = {x : x for x in graph.nodes()}

    cut_edges = set( graph.graph["ordered_edges"])
    non_cut_edges = set([])

    samples = []

    for i in range(steps):
        non_cut_edges, cut_edges = step(graph, cut_edges, non_cut_edges, temperature)
        if contradictions(graph, cut_edges) == 0:
            samples.append([copy.deepcopy(cut_edges),copy.deepcopy(non_cut_edges), copy.deepcopy(graph.graph["coloring"])])

    print(len(samples))

    return samples