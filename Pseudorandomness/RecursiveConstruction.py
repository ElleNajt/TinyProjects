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

def viz(graph):
    pos = {x : graph.nodes[x]["pos"] for x in graph.nodes()}
    nx.draw(graph, pos, node_size = 1, width =2, cmap=plt.get_cmap('magma'))

def create_layered_digraph(n = 4, d = 2**3):

    n = 4
    l = 3
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
        
    viz(graph)
    
    return graph 

def find_min_paths(graph,s,t):
    
    return 0
    
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
        return (a*x + b) % r
    
    return hash_function
    
def pure_hashing_assign_weights(graph):
    #Via Pure Hashing Approach
    

    num_nodes = len(graph.nodes())
    r_value = nextprime(num_nodes**6)
    
    l = graph.graph["num_layers"]
    n = graph.graph["width"]
    
    for k in range(l):
        hash_function = new_hash_function(r_value)
        #The middle layers over all blocks of depth 2^{k+1}.  L = Union_{odd i \in [2^{l - k }]} V_{i 2^k}
        #Run through L 
        L = []
        for i in range(2**(l - k)):
            if i % 2 == 1:
                for j in range(n):
                    L.append( (j, i * (2**k)))
        print(L)
        for x in L:
            graph.nodes[x]["weight"] += hash_function(graph.nodes[x]["label"])
                
            
            
    return graph

graph = create_layered_digraph()
graph = pure_hashing_assign_weights(graph)