# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:48:24 2019

@author: Lorenzo
"""


import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import Facefinder
import copy
def viz(graph, names = True):


    nx.draw(graph, pos=nx.get_node_attributes(graph, 'coord'), node_size = 10, width = 1, cmap=plt.get_cmap('jet'), with_labels=names)



#We're going to construct the gadget that is the recursive layer

def convert_label(edge):
    #this will convert [ak, bk]to c'k
    
    adjacent_vertices = set ( [edge[0][0], edge[1][0]])
    for x in ['a', 'b', 'c']:
        if x not in adjacent_vertices:
            label = x
    level = edge[0][1:]
    return str(label) + 'O' + str(level)
    
def layer_label(string, level):
    
    label = string[0]
    return label + str(level)

def make_layer(names,level):
    layer = nx.cycle_graph(3)
    node_list = list(layer.nodes())
    for x in node_list:
        #vector = [math.cos( ( ( (-1)**(level + 1) )* x * 4 + (level % 2) ) * math.pi / 6), math.sin( ( ( (-1)**(level+ 1)) *x * 4 + (level % 2) )  * math.pi / 6)]
        vector = [math.cos( ( x * 4 + (level % 2) ) * math.pi / 6), math.sin(  (x * 4 + (level % 2) )  * math.pi / 6)]
        
        layer.node[x]["coord"] = (vector[0] / math.sqrt((level+1)), vector[1] / math.sqrt((level+1)))
        layer.add_node(str(names[x]))
        layer.node[str(names[x])]["coord"] = (vector[0] / (math.sqrt(level)), vector[1]/ (math.sqrt(level)))
        layer.add_edge(x, str(names[x]))
        layer = nx.relabel_nodes(layer, {x: layer_label(names[x], level)})
    
    if (level % 2) == 1:
        layer.graph["Cedges"] = [ ['b' + str(level), 'c' + str(level)],  ['c' + str(level), 'a' + str(level)], ['a'+ str(level), 'b' + str(level)] ]
    if (level % 2) == 0:
        layer.graph["Cedges"] = [['a'+ str(level), 'b' + str(level)],  ['b' + str(level), 'c' + str(level)],  ['c' + str(level), 'a' + str(level)] ]
        
    return layer


def subdivide(R):
    for e in R.graph["Cedges"]:
        
        R.add_path( [ e[0], convert_label(e), e[1]])
        R.remove_edge(e[0], e[1])
        
    layer = make_layer([convert_label(edge) for edge in R.graph["Cedges"]], R.graph["level"] + 1)
    
    F = nx.compose(layer, R)
    for x in layer.nodes():
        F.node[x]["coord"] = layer.node[x]["coord"]
    #viz(F)
    
    #nx.draw(F, with_labels = True)

    F.graph["Cedges"] = list(layer.graph["Cedges"])
    F.graph["level"] = R.graph["level"] + 1
    R = F
    return(R)


def construct_gadget(depth):
    R = nx.cycle_graph(3)
    for x in R.nodes():
        R.node[x]["coord"] = (math.cos( x * 2 * math.pi / 3), math.sin( x * 2 * math.pi / 3))
    R = nx.relabel_nodes(R, {0: 'a0', 1: 'b0', 2: 'c0'})
    
    R.graph["Cedges"] =[('a0', 'b0'),('b0', 'c0'), ('a0', 'c0')]
    R.graph["level"] = 0


    while R.graph["level"] <= depth:
        R = subdivide(R)
    for x in R.nodes():
        R.node[x]["pos"] = R.node[x]["coord"]
    return R

def construct_gadget_with_clasp(k):
    graph = construct_gadget(k)
    graph.add_node("S")
    graph.add_edge("S", 'a0')
    graph.add_edge("S", 'b0')
    graph.add_edge("S", 'c0')
    graph.node["S"]["pos"] = [6,9]
    graph.node["S"]["coord"] = [6,9]
    return graph

graph = construct_gadget(3)

graph = Facefinder.compute_rotation_system(graph)
graph = Facefinder.compute_face_data(graph) 
#print(len(graph.graph["faces"]))

dual_R = Facefinder.restricted_planar_dual(graph)

#Facefinder.draw_with_location(dual_R)


def convert_to_edges(graph, nodes):
    
    H = nx.induced_subgraph(graph,nodes)
    return list(H.edges())

def check_simple_cycle(graph, edges):
    edge_list = []
    for x in edges:
        edge = []
        for e in x:
            edge.append(e)
        edge = tuple(edge)
        edge_list.append(edge)
    H = nx.edge_subgraph(graph, edge_list)
    if len(H) == 0:
        return True
    if not nx.is_connected(H):
        return False
    if np.max( list( dict(nx.degree(H)).values())) > 2:
        return False
    return True

def add(edges1, edges2):
    output = []
    #Have to write your own symmetric difference between of ordering of edges
    for e in edges1:
        if not e in edges2:
            output.append(e)
    for e in edges2:
        if not e in edges1:
            output.append(e)    
    return frozenset(output)

def cycle_basis(graph):
    vertex_listed = nx.cycle_basis(graph)
    #NX gives the cycle basis as a list of vertices, not by edges

def enumerate_simple_cycles(graph):
    #This enumerate the simple cycles of the input graph
    #The input graph must be 2 connected in order for this to work
    #Also, the input graph must come with a planar embedding, given by (x,y) 
    #coordinates of the nodes. We can get around this, by constructin a cycle 
    #basis using linear algebra, rather than the embedding... but for now its ok
    graph = Facefinder.compute_rotation_system(graph)
    graph = Facefinder.compute_face_data(graph) 
#print(len(graph.graph["faces"]))

    dual_R = Facefinder.restricted_planar_dual(graph)

    basis = [convert_to_edges(graph, x) for x in dual_R.nodes()]
    #basis = cycle_basis(graph)
    
    set_basis = []
    for b in basis:
        set_basis.append( frozenset( [ frozenset(x) for x in b]))
    basis = set_basis
    metagraph = nx.Graph()
    metagraph.add_node(frozenset(basis[0]))
    wet = set([frozenset(basis[0])])
    while len(wet) > 0:
        #print(len(metagraph))
        wet_list = list(wet)
        wet = set()
        for x in wet_list:
            neighbors = [ add(x,b) for b in basis if check_simple_cycle(graph, add(x,b))]
            set_neighbors = [frozenset(x) for x in neighbors]
            #print(neighbors)
            #print([len(x) for x in neighbors])
            new_wets = [x for x in set_neighbors if x not in metagraph.nodes()]
            #print(len(new_wets), "wets")
            for j in set_neighbors:
                metagraph.add_node(j)
            wet = wet.union(new_wets)
    
    print(len(metagraph))

def connectivity():
    
    for i in range(40):
        graph = construct_gadget_with_clasp(i)
        print(nx.node_connectivity(graph))
        


def counts():
    #For counting the simple cycles
    
    for i in range(6):
        #enumerate_simple_cycles(construct_gadget(i))
        graph = construct_gadget(i)
        G = nx.DiGraph()
        for x in graph.nodes():
            G.add_node(x)
        for e in graph.edges():
            G.add_edge(e[0],e[1])
            G.add_edge(e[1],e[0])
        S = list(nx.simple_cycles(G))
    
        print( (len(S) - len(list(graph.edges())) )/2)
        #print( len(with_S) / 2)
    
    
    #For counting the simple boundary links:
    
    for i in range(-1,6):
        #enumerate_simple_cycles(construct_gadget(i))
        graph = construct_gadget_with_clasp(i)
        G = nx.DiGraph()
        for x in graph.nodes():
            G.add_node(x)
        for e in graph.edges():
            G.add_edge(e[0],e[1])
            G.add_edge(e[1],e[0])
        S = list(nx.simple_cycles(G))
        with_S = [x for x in S if 'S' in x and len(x) > 2 ] 
        print( len(with_S) / 2)
    
    
    
    
    
    
    
    for k in range(2,3):
        graph = nx.grid_graph([k,k])
        for x in graph.nodes():
            
            graph.node[x]["pos"] = np.array([x[0], x[1]])
        enumerate_simple_cycles(graph)
        
        np.max(dict(nx.degree(dual_R)).values())
    
    #Can we fix the faces to be convex?
    
    #The nodes of the dual give the cycle basis -- this is enough to enumerate the metagraph 

def translate_graph(G, vector):
    for x in G.nodes():
        pos = G.node[x]["coord"] 
        G.node[x]["coord"] = [pos[0] + vector[0], pos[1]+ vector[1]]
    return G
        
def augment(graph, depth):
     #graph must be cubic. This replaces the vertices of graph with depth-depth gadgets
     #graph should be planar with "coord" node attributes
     #there should be enough room between nodes to fit in the gadgets
     RG = nx.Graph()
     
     #First we are going to replace every vertex x of graph with a triangle labeled by xa0, xb0, xc1...
     
     
     
     #now we compose the gadgets for each of these...
     
     #which takes coordinate data precedence?
     
     graph = nx.complete_graph(4)
     location = {0 : [1,0], 1: [-1,0], 2: [0,1], 3: [0,2]}
     for x in graph.nodes():
         graph.node[x]["coord"] = location[x]
     
     node_list = list(graph.nodes())
     
     
     for x in node_list:
         edges = list(graph.edges(x))
         names = {0 : 'a', 1: 'b', 2: 'c'}
         new_vertices = []
         for i in range(3):
             e = edges[i]
             a = graph.node[e[0]]["coord"]
             b = graph.node[e[1]]["coord"]
             if x == 2:
                 mp = [(a[0] + b[0])/2, (a[1] + b[1])/2]
             if x != 2:
                 if 2 not in [ e[0], e[1]]:
                     mp = [(a[0] + b[0])/2, (a[1] + b[1])/2]                    
                 if 2 in [ e[0], e[1] ]:
                     this = graph.node[x]["coord"]
                     two = graph.node[2]["coord"]
                     mp = [(.1* this[0] + .9 * two[0]), (.1* this[1] + .9 * two[1])]
                
             new_vertex = str(x) + names[i]+ '0'
             graph.add_path([e[0], new_vertex, e[1]])
             graph.node[new_vertex]["coord"] = mp
             new_vertices.append(new_vertex)
             
         for e in new_vertices:
             for f in new_vertices:
                 graph.add_edge(e,f)
         
         graph.remove_node(x)
     
     v = graph.node['3b0']["coord"]
     graph.node['3b0']["coord"] = [ v[0], v[1] + 2]
     v = graph.node['3a0']["coord"]
     graph.node['3a0']["coord"] = [ v[0], v[1] + 2]
     
     for v in graph.nodes():
         coordinate = graph.node[v]["coord"]
         graph.node[v]["coord"] = [10*coordinate[0], 10*coordinate[1]]
     #v = graph.node['3c0']["coord"]
     #graph.node['3c0']["coord"] = [ v[0], v[1] + 2]
     
    # viz(graph)
    
     RG = graph
     depth = 2
     for x in node_list:
         gadget = construct_gadget(depth)
         gadget = translate_graph(gadget, location[x])
         for y in gadget.nodes():
             nx.relabel_nodes(gadget, {y : str(x) + str(y) })
         RG = nx.compose(RG, gadget)
     viz(RG)

    #How to add edges ? A cyclic orientation would help...
    #Or, first replace every vertex of G by a triangle with $a0, b0, c0$, and then compose 
        
