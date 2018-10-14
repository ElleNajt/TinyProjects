# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 19:39:52 2018

@author: Temporary
"""

###This will take a graph that (I know is) planar along with position data on the nodes, and construct face data.

import networkx as nx
import numpy as np



def compute_rotation_system(graph):
    #Graph nodes must have "pos"
    for v in graph.nodes():
        graph.node[v]["pos"] = np.array(graph.node[v]["pos"])
    
    for v in graph.nodes():
        locations = []
        neighbor_list = list(graph.neighbors(v))
        for w in neighbor_list:
            locations.append(graph.node[w]["pos"] - graph.node[v]["pos"])
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)
        #sorted_neighbors = [x for _,x in sorted(zip(angles, neighbor_list))]
        rotation_system = {}
        for i in range(len(neighbor_list)):
            rotation_system[neighbor_list[i]] = neighbor_list[(i + 1) % len(neighbor_list)]
        graph.node[v]["rotation"] = rotation_system
    return graph




def compute_face_data(graph):
    #graph must already have a rotation_system
    faces = []
    #faces will stored as sets of vertices

    for e in graph.edges():
        face = set([e[0], e[1]])
        last_point = e[1]
        current_point = graph.node[e[1]]["rotation"][e[0]]
        next_point = current_point
        while next_point != e[0]:
            face.add(current_point)
            next_point = graph.node[current_point]["rotation"][last_point]
            last_point = current_point
            current_point = next_point
        faces.append(frozenset(face))
        
    #Insert remove outer face
    graph.graph["faces"] = set(faces)
    return graph
            


def face_refine(graph):
    #graph must already have the face data computed
    
    for face in graph.graph["faces"]:
        graph.add_node(face)
        location = np.array([0,0])
        for v in face:
            graph.add_edge(face, v)
            location += graph.node[v]["pos"]
        graph.node[face]["pos"] = location / len(face)
    return graph

def refine(graph):
    graph = compute_rotation_system(graph)
    graph = compute_face_data(graph)
    graph = face_refine(graph)
    return graph

def draw_with_location(graph):
#    for x in graph.nodes():
#        graph.node[x]["pos"] = [graph.node[x]["X"], graph.node[x]["Y"]]

    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 200/k, width = .5, cmap=plt.get_cmap('jet'))
    
graph = nx.grid_graph([2,2])
for x in graph.nodes():
    
    graph.node[x]["pos"] = np.array([x[0], x[1]])

graph = refine(graph)

draw_with_location(graph)


