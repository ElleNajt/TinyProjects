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
import random


def dist(v, w):
    return np.linalg.norm(np.array(v) - np.array(w))


def face_contained_in_disc(coord, r):
    # given the center of the 1x1 face, determine if it is contained in the open unit disc.
    for a in [-.5, .5]:
        for b in [-.5, .5]:
            if np.linalg.norm([coord[0] + a, coord[1] + b]) >= r:
                return False
    return True


def edges_of_dual_face(coord):
    # takes in the center of the dual face as an array of coordinates, and returns the 4 edges.
    vertices = []
    for a in [-.5, .5]:
        for b in [-.5, .5]:
            vertices.append( (coord[0]+ a + (coord[1] + b)*1j))
    edges = set([])
    for x in vertices:
        for y in vertices:
            if dist(x,y) == 1:
                edges.add(frozenset([x,y]))
    a = coord[0]
    b = coord[1]
    edges = [((a + .5, b + .5), (a - .5, b+ .5)), ((a - .5, b + .5), (a - .5, b - .5)), ( ( a - .5, b - .5), (a + .5, b - .5)), ( ( a + .5, b - .5), (a + .5 , b + .5)) ]
    int_edges = [tuple([tuple([int(x) for x in a]) for a in e]) for e in edges]

    return vertices, int_edges


def create_disc_graph(r):
    # Creates the graph representing the intersection of $\mathbb{Z}^2$ with a disc of radius r

    grid = nx.grid_graph([4*r, 4*r])
    dual_grid = nx.grid_graph([4*r, 4*r])
    relabels = {}
    for v in grid.nodes():
        grid.node[v]["coord"]= [ v[0]/r - 2, v[1]/r - 2]
        dual_grid.node[v]["coord"] = [ ( v[0]+ .5)/r - 2, (v[1] + .5)/r - 2]
        relabels[v]= str(grid.node[v]["coord"])
    #
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


def integral_disc(r):
    # Takes in r (should be an integer, and returns the integral points in the disc of radius r + .5
    grid = nx.grid_graph([4 * r, 4 * r])
    dual_grid = nx.grid_graph([4 * r, 4 * r])
    relabels = {}
    dual_relabels = {}
    for v in grid.nodes():
        grid.node[v]["coord"] = (v[0] - 2*r, v[1] - 2*r)
        dual_grid.node[v]["coord"] = ((v[0] + .5) - 2*r, (v[1] + .5) - 2*r)
        relabels[v] = grid.node[v]["coord"]
        dual_relabels[v] = dual_grid.node[v]["coord"]
    grid = nx.relabel_nodes(grid, relabels)
    dual_grid = nx.relabel_nodes(dual_grid, dual_relabels)
    intersection_nodes = [v for v in grid.nodes() if np.linalg.norm(grid.node[v]["coord"]) < r + .5]
    intersection_faces = [v for v in dual_grid.nodes() if face_contained_in_disc(dual_grid.node[v]["coord"], r + .5)]
    # Now this gives exactly the dual verites whose facesare in the unit disc.

    disc_graph = nx.subgraph(grid, intersection_nodes)
    dual_disc_graph = nx.subgraph(dual_grid, intersection_faces)

    # Now we need to add code so that each dualface can report its edges

    disc_graph.graph["dual"] = dual_disc_graph

    disc_graph.graph["plus_one"] = (r, 0)
    disc_graph.graph["minus_one"] = (-r,0)
    disc_graph.graph["scale"] = r + .5
    # These two are the nearest points on the graph to $-1$ and $+1$.

    # This is a little funny in the integral case... so we set the disc to size r + .5
    return disc_graph


def viz(T, path):
    k = 20

    for x in T.nodes():
        if x in path:
            T.node[x]["col"] = 0
        else:
            T.node[x]["col"] = 1
    values = [T.node[x]["col"] for x in T.nodes()]

    nx.draw(T, pos=nx.get_node_attributes(T, 'coord'), node_size = 1, width = .5, cmap=plt.get_cmap('jet'),  node_color=values)

def viz_edge(T, edge_path):
    k = 20

    values = [1 - int((x in edge_path) or ((x[1], x[0]) in edge_path)) for x in T.edges()]

    nx.draw(T, pos=nx.get_node_attributes(T, 'coord'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'),  edge_color=values)


def map_up(point, r):
    # This takes the point and scale it down by r + .5 (we are using integral disc for the markov chain)
    # This sends a point to the upperhalf plane via the map $g(z) = i ( 1+ z)/(1 - z)$
    # In this, -1 is sent to 0, and 1 is sent to infinity. So these are the marked points.
    complex_point = (point[0] + point[1]*1j)/(r + .5)
    new_value = 1j * (1 + complex_point)/(1 - complex_point)
    return [new_value.real, new_value.imag]

def test_create_and_map():
    disc = integral_disc(20)
    path = initial_path(disc)
    viz(disc, path)

    for v in D.nodes():
        D.node[v]["coord"]=  map_up(D.node[v]["coord"])

    viz(D)
    # THere are some serious distortions here!This is going to be an issue probably, b
    # ut maybe we can avoid it by choosing discs that are near to the origin.


def initial_path(disc_graph):
    path = [disc_graph.graph["minus_one"]]
    current = path[0]
    i = 0
    while (current != disc_graph.graph["plus_one"]) and (i <= 3*disc_graph.graph["scale"]):
        new = (current[0] + 1, current[1])
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

def convert_node_sequence_to_edge(path):
    # For analysis and visualization, it will be better to use the edges
    edge_path = []
    for i in range(len(path) - 1):
        edge_path.append((path[i], path[i+1]))
    return edge_path


def try_add_faces(path, edges):
    # cases depending on the size of the intersection -- a little sloppier but will be faster?
    # Algorithm -- walk down path until interect vertics of the edges...
    # delete the edges from *edges* as you walk down the path,
    # then past in teh remaining edges in the appropriate pace.

    # Note -- on a square  think self avoiding is all we need to check after a proposal,
    # but on a graph with higher degree faces, you will need to check that this swap doesn't disconnect the walk.

    # Disconnects can still happen here... e.g. adding a blcok in the long hallway. But this code takes care of that
    # situation, since that will break self avoiding -- it makes changes to the path as a function fo time
    square = nx.Graph()
    square.add_edges_from(edges)

    vertices = set([x[0] for x in edges] + [x[1] for x in edges])
    i = 0
    while path[i] not in vertices:
        i += 1
        if i == len(path):
            return False
            #return False
            # This is the case when the face is disjoint.
    j = i + 1
    if j >= len(path):
        # This is the case that i is the endpoint of the path, i.e. that the path touches the face at the end
        return False
    old_path_through_square = set([])
    while ((j < len(path)) and (path[j] in vertices)):
        # It might be ht ecase that exist_node is the last point of the path
        old_path_through_square.add(path[j])
        j += 1

    exit_node = path[j-1]
    if exit_node == path[i]:
        # This is the case of a corner touching
        return False

    new_path = []
    for t in range(i):
        new_path.append(path[t])

    if i == 0:
        # Exception when the block is at the head
        t = -1

    current = path[t+1]
    last = current
    while current != exit_node:
        new_path.append(current)
        neighs = set(square.neighbors(current))
        for x in old_path_through_square:
            if x in neighs:
                neighs.remove(x)
        if last in neighs:
            neighs.remove(last)
        last = current
        if len(neighs) == 0:
            current = exit_node
        else:
            current = list(neighs)[0]
    for m in range(j - 1, len(path)):
        new_path.append(path[m])

    return new_path


def step(disc, path):
    dual = disc.graph["dual"]
    face = random.choice(list(dual.nodes()))
    vertices, edges = edges_of_dual_face(dual.node[face]["coord"])

    new_path = try_add_faces(path, edges)

    while new_path is False:
        face = random.choice(list(dual.nodes()))
        vertices, edges = edges_of_dual_face(dual.node[face]["coord"])
        new_path = try_add_faces(path, edges)
    if check_self_avoiding(new_path):
        path = new_path

    return path

def test_run(disc, path):

    for i in range(1000):
        try:
            path = step(disc, path)
        except:
            print("failed")
    edge_path = convert_node_sequence_to_edge(path)
    viz_edge(disc, edge_path)


