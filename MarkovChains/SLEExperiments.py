# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:47:44 2019

@author: Temporary
"""

#What we'll do is simulate the Markov chain on the intersection with the disc.
#Then we can compute the explicit mapping out fomulas...

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import Facefinder
import random


def dist(v, w):
    return np.linalg.norm(np.array(v) - np.array(w))


def face_contained_in_disc(coord):
    #given the center of the 1x1 face, determine if it is contained in the open unit disc.
    for a in [-.5, .5]:
        for b in [-.5, .5]:
            if np.linalg.norm([coord[0] + a, coord[1] + b]) >= 1:
                return False
    return True


def edges_of_dual_face(coord):
    #takes in the center of the dual face as an array of coordinates, and returns the 4 edges.
    vertices = []
    for a in [-.5, .5]:
        for b in [-.5, .5]:
            vertices.append( (coord[0]+ a + (coord[1] + b)*1j))
    edges = set([])
    for x in vertices:
        for y in vertices:
            if dist(x,y) == 1:
                edges.add(frozenset([x,y]))
    a= coord[0]
    b = coord[1]
    edges = [ ((a + .5, b + .5), (a - .5, b+ .5)), ((a - .5, b + .5), (a - .5, b - .5)), ( ( a - .5, b - .5), (a + .5, b - .5)), ( ( a + .5, b - .5), (a + .5 , b + .5)) ]
    int_edges= [ [[ int(x) for x in a] for a in e] for e in edges]
    
    return vertices, edges


def disc_graph(r):
    #C reates the graph representing the intersection of $\mathbb{Z}^2$ with a disc of radius r

    grid = nx.grid_graph([4*r, 4*r])
    dual_grid = nx.grid_graph([4*r, 4*r])
    relabels = {}
    for v in grid.nodes():
        grid.node[v]["coord"]= [ v[0]/r - 2, v[1]/r - 2]
        dual_grid.node[v]["coord"] = [ ( v[0]+ .5)/r - 2, (v[1] + .5)/r - 2]
        relabels[v]= str(grid.node[v]["coord"])
    # nx.relabel_nodes(grid, relabels)
    # Actually don't, because it is useful to have integral names
    intersection_nodes = [v for v in grid.nodes() if np.linalg.norm(grid.node[v]["coord"]) < 1]
    intersection_faces = [v for v in dual_grid.nodes() if face_contained_in_disc(dual_grid.node[v]["coord"])]
    # Now this gives exactly the dual verites whose facesare in the unit disc.
    
    disc_graph = nx.subgraph(grid, intersection_nodes)
    dual_disc_graph = nx.subgraph(dual_grid, intersection_faces)
    
    # Now we need to add code so that each dualface can report its edges

    disc_graph.graph["dual"] = dual_disc_graph
    minus_one = list(disc_graph.nodes())[0]
    plus_one = list(disc_graph.nodes())[0]
    for v in disc_graph.nodes():
        if dist(disc_graph.node[v]["coord"], [-1,0]) < dist(disc_graph.node[minus_one]["coord"], [-1,0]):
            minus_one = v
        if dist(disc_graph.node[v]["coord"], [1,0]) < dist(disc_graph.node[plus_one]["coord"], [1,0]):
            plus_one = v
    print(disc_graph.node[plus_one]["coord"], disc_graph.node[minus_one]["coord"])
    disc_graph.graph["plus_one"] = plus_one
    disc_graph.graph["minus_one"] = minus_one
    disc_graph.graph["scale"] = r
    # These two are the nearest points on the graph to $-1$ and $+1$.
    return disc_graph


def viz(T):
    k = 20
    n = 10
    nx.draw(T, pos=nx.get_node_attributes(T, 'coord'), node_size = 200/k, width = .5, cmap=plt.get_cmap('jet'))


def map_up(point):
    # This sends a point to the upperhalf plane via the map $g(z) = i ( 1+ z)/(1 - z)$
    # In this, -1 is sent to 0, and 1 is sent to infinity. So these are the marked points.
    complex_point = point[0] + point[1]*1j
    new_value = 1j* ( 1 + complex_point)/(1 - complex_point)
    return [new_value.real, new_value.imag]

def test_create_and_map():
    D = disc_graph(20)
    viz(D)
    path = initial_path(D)
    
    for v in D.nodes():
        D.node[v]["coord"]=  map_up(D.node[v]["coord"])
    
    viz(D)
    # THere are some serious distortions here!This is going to be an issue probably, but maybe we can avoid it by choosing discs that are near to the origin.


def initial_path(disc_graph):
    path = [disc_graph.graph["minus_one"]]
    current = path[0]
    i = 0
    while (current != disc_graph.graph["plus_one"]) and (i <= 3*disc_graph.graph["scale"]):
        new = [current[0] + 1, current[1]]
        current = new
        path.append(new)
        i += 1
    return path


def check_self_avoiding(path):
    # Returns true if the path is self avoiding
    i = 0
    length = len(path)
    while i <= length - 1:
        for x in path[i + 1 : length]:
            if x == path[i]:
                return False
        i += 1
    return True


def try_add_faces(path, edges):
    # cases depending on the size of the intersection -- a little sloppier but will be faster?
    # Algorithm -- walk down path until interect vertics of the edges...
    # delete the edges from *edges* as you walk down the path,
    # then past in teh remaining edges in the appropriate pace.
    
    # Note -- on a square  think self avoiding is all we need to check after a proposal,
    # but on a graph with higher degree faces, you will need to check that this swap doesn't disconnect the walk.
    
    square = nx.Graph()
    square.add_edges_from(edges)
    
    vertices = set([x[0] for x in edges] + [x[1] for x in edges])
    i = 0
    while path[i] not in vertices:
        i += 1
        if i == len(path):
            return False
            # This is the case when the face is disjoint.
    j = i + 1
    old_path_through_square = set([])
    while path[j] in vertices:
        old_path_through_square.add(path[j])
        j += 1
        
    exit_node = path[j-1]

    new_path = []
    for t in range(i+1):
        new_path.append(path[t])
        
    current = path[t]
    last = current
    while current != exit_node:
        new_path.append(current)
        neighs = set(square.neighbors(current))
        for x in old_path_through_square:
            neighs.remove(x)
        neighs.remove(last)
        last = current
        current = list(neighs)[0]
    for m in range(j - 1, len(path)):
        new_path.append(path[m])

    return new_path


def step(disc_graph, path):
    dual = disc_graph.graph["dual"]
    
    face = random.choice(list(dual.nodes()))
    vertices, edges = edges_of_dual_face(dual.node[face]["coord"])
    new_path = add_faces(path, edges)
    if check_self_avoiding(new_path):
        path = new_path
    
    return path
    
        
