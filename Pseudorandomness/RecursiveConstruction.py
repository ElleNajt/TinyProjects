# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:42:31 2020

@author: lnajt
"""

import itertools
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import secrets
import sympy
from sympy import nextprime

'''
Things to experiment with:
    
    1) Expander Steps on Hashes
    2) Random shifts
    3) Edge Replacement Gadgets
    
Ways to validate experiments:
    
    1) Whether min-path is unique. ()
'''

def viz(graph):
    pos = {x : graph.nodes[x]["pos"] for x in graph.nodes()}
    col = [graph.nodes[x]["weight"] for x in graph.nodes()]
    nx.draw(graph, pos, node_color =col, node_size = 100, width =.5, cmap=plt.get_cmap('hot'))

def create_layered_digraph(n = 4, l = 3):

    d = 2**l
    
    layered_nodes = itertools.product(range(n), range(d + 1))
    
    graph = nx.Graph()
    graph.add_nodes_from(layered_nodes)
    
    
    for x in graph.nodes():
        #print(x)
        graph.nodes[x]["pos"] = np.array([x[0], x[1]])
        
         
    position = {x : graph.nodes[x]["pos"] for x in graph.nodes()}
    #nx.draw(graph, pos = position, node_size = 1, width =2, cmap=plt.get_cmap('magma'))
    
    graph.graph["num_layers"] = l
    graph.graph["width"] = n
    for l in range(d):
        for k in range(n):
            for p in range(4):
                u = random.choice(list(range(n)))
                graph.add_edge( (k,l), (u, l+1))
                
    i = 0
    for x in graph.nodes():
        #print(x)
        graph.nodes[x]["pos"] = np.array([x[0], x[1]])
        graph.nodes[x]["label"] = i
        i += 1
        graph.nodes[x]["weight"] = 0
        
    #viz(graph)
    graph.graph["s"] = (0,0)
    graph.graph["t"] = (0,d)
    return graph 

def find_min_paths(graph):
    ##Checks whether there is a unique min weight path from s to t. Algorithm tests the disambiguation requirement layer by layer. [TODO: Maybe another algorithm for testing min-unique st-path would lead to a different disambiguation requirement? Is it possible to use the randomness adaptively, by finding where the min-uniqueness breaks and then rerolling the hash function, e.g. by taking a step on the expander?]
    #E.g. compare Dijkstra, Bellman-Ford, many more here: https://networkx.github.io/documentation/stable/reference/algorithms/shortest_paths.html
    s = graph.graph["s"]
    t = graph.graph["t"]
    if nx.has_path(graph, s,t):
        paths = list(nx.all_shortest_paths(graph, s,t, weight = "weight"))
        return paths
    else:
        return []
    
def new_hash_function(r):
    # Produces a pairwise independent hash function h : [m]-> [r] 
    # bits = secrets.randbits(10) #pass through bin if you want bits
    
    #We'll assume r > m, and that r is prime.
    
    #NB: Dropping the modulus keeps the collision probability down, although it's no longer uniform, and the range is different (r^2 instead of r). 
    
    # This is similar to shifting? What happens for random shifts? I.e. multiplication in N(atural numbers) against a random number randbelow(r). TODO.
    if not sympy.isprime(r):
        return False
    
    a = secrets.randbelow(r)
    b = secrets.randbelow(r)
    
    def hash_function(x):
        return (a*x + b) % r, r
    
    return hash_function

def torus_step(hash_function):
    b = hash_function(0)[0]
    a = hash_function(1)[0] - b
    r = hash_function(0)[1]
    neighbors = [(a,b), (a + 1, b), (a, b + 1), (a, a + b), ( -1 * b , a)]
    
    step = secrets.choice(neighbors)
    
    a = step[0]
    b = step[1]
    
    
    def hash_function(x):
        return (a*x + b) % r, r
    
    return hash_function
    
    
def pure_hashing_assign_weights(graph, hash_once = False, expander_hash = False):
    #Via Pure Hashing Approach
    

    num_nodes = len(graph.nodes())
    r_value = nextprime(num_nodes**6)
    
    l = graph.graph["num_layers"]
    n = graph.graph["width"]
    
    if hash_once == True or (hash_once == False and expander_hash == True):
        hash_function = new_hash_function(r_value)
    
    for k in range(l):
        if hash_once == False:
            if expander_hash == False:
                hash_function = new_hash_function(r_value)
            if expander_hash == True:
                hash_function = torus_step(hash_function)
        #The middle layers over all blocks of depth 2^{k+1}.  L = Union_{odd i \in [2^{l - k }]} V_{i 2^k}
        #Run through L 
        L = []
        for i in range(2**(l - k)):
            if i % 2 == 1:
                for j in range(n):
                    L.append( (j, i * (2**k)))
        #print(L)
        for x in L:
            graph.nodes[x]["weight"] += hash_function(graph.nodes[x]["label"])[0]
                
    return graph


def node_weights_to_edge_weights(graph):
    #Updates the edge weights based on node weights. For 
    for e in graph.edges():
        graph.edges[e]["weight"] = graph.nodes[e[0]]["weight"] + graph.nodes[e[1]]["weight"]
        
    return graph

uniques = 0
for i in range(1000):
    graph = create_layered_digraph(10,8)
    graph = pure_hashing_assign_weights(graph, hash_once = False, expander_hash = True)
    #viz(graph)
    graph = node_weights_to_edge_weights(graph)
    paths = find_min_paths(graph)
    if len(paths) == 1:
        uniques += 1
print(uniques / 1000)
viz(graph)

'''Observations:
   For Hash-Once = True, on (10,8), chance of min-unique drops to zero. 
   For Hash-Once = False, on (10,8), chance of min-unique is around .8.
   For Torus-Expander, on (10,8), chance of min-unique is around .25
    
'''