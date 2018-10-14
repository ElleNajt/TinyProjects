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


def cycle_around_face(graph, e):
    face = set([e[0], e[1]])
    last_point = e[1]
    current_point = graph.node[e[1]]["rotation"][e[0]]
    next_point = current_point
    while next_point != e[0]:
        face.add(current_point)
        next_point = graph.node[current_point]["rotation"][last_point]
        last_point = current_point
        current_point = next_point
    return face


def compute_face_data(graph):
    #graph must already have a rotation_system
    faces = []
    #faces will stored as sets of vertices

    for e in graph.edges():
        #need to make sure you get both possible directions for each edge..
        
        face = cycle_around_face(graph, e)
        faces.append(frozenset(face))
        
        #how to detect if face is the unbounded face...
    

        face = cycle_around_face(graph, [ e[1], e[0]])
        faces.append(frozenset(face))
        
        
        
    #Insert remove outer face
    print("reminder that you still have the outer face -- which might be okay, because maybe it also counts the inner face... ifthe graph is a single fae")
    graph.graph["faces"] = set(faces)
    return graph
            


def face_refine(graph):
    #graph must already have the face data computed
    #this adds a vetex in the middle of each face, and connects that vertex to the edges of that face...
    
    for face in graph.graph["faces"]:
        graph.add_node(face)
        location = np.array([0,0]).astype("float64")
        for v in face:
            graph.add_edge(face, v)
            location += graph.node[v]["pos"].astype("float64")
        graph.node[face]["pos"] = location / len(face)
    return graph

def refine(graph):
    graph = compute_rotation_system(graph)
    graph = compute_face_data(graph)
    graph = face_refine(graph)
    return graph

def depth_k_refine(graph,k):
    graph.name = graph.name + str("refined_depth") + str(k)
    for i in range(k):
        graph = refine(graph)
    return graph

def draw_with_location(graph):
#    for x in graph.nodes():
#        graph.node[x]["pos"] = [graph.node[x]["X"], graph.node[x]["Y"]]

    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 0, width = .5, cmap=plt.get_cmap('jet'))
# 
m= 2
graph = nx.grid_graph([m,m])
graph.name = "grid_size:" + str(m)
for x in graph.nodes():
    
    graph.node[x]["pos"] = np.array([x[0], x[1]])

graph = depth_k_refine(graph,10)

draw_with_location(graph)


