# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:28:13 2020

@author: lnajt
"""

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import random
import Facefinder

def step_num(partition):
                parent = partition.parent

                if not parent:
                    return 0

                return parent["step_num"] + 1


def nearest_point(point, r = 1):
    #This finds the nearest point to point on the circle of radius r.
    
    norm = np.linalg.norm(point)
    np_point = np.array(point)
    normalized_point = (r / norm) * np_point
    
    return np_point


def rotate(x,y,p):
    #rotates y to 1, and x and p come along for the ride.
    complex_y = (y[0] + y[1]*1j)
    angle = np.angle(complex_y)
    
    rotation_matrix = [ [ np.cos( angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    
    return np.matmul(rotation_matrix, x) , np.matmul(rotation_matrix, y) , np.matmul(rotation_matrix, p)

def map_up(point):
    # This sends a point to the upperhalf plane via the map $g(z) = i ( z + 1)/(-z + 1 )$
    # In this, -1 is sent to 0, and 1 is sent to infinity. So these are the marked points.
    complex_point = (point[0] + point[1]*1j)
    new_value = 1j * (complex_point + 1)/(-1*complex_point + 1)
    return [new_value.real, new_value.imag]

def map_down(point):
    # Inverse of map up. inverse of $g$ is : $h(z) =  z - i / (z + i)
    complex_point = (point[0] + point[1]*1j)
    new_value =  (complex_point - 1j )/( complex_point + 1j)
    return [new_value.real, new_value.imag]


def test_up_down():
    
    for v in [ [0,0], [-1, 0], [.1,.2]]:
        print(v, map_down(map_up(v)))

def conformal_automorphism(x,y,p):
    #x,y,a,b are 4 points on the 1 circle.
    #applies a (standardized) conformal automorphism of the disc mapping (x,y) -> (-1,1)
    #to the point p
    
    x, y, p = rotate(x,y,p)
    #rotates y to 1, and x and p come along for the ride.

    p_new = map_down( map_up(p) - map_up(x))
    #mpa_down( map_up(z) - map_up(x)) causes x to move to -1, and keeps 1 where it is.
    
    return p_new

def conformally_align(path):
    
    #Given a sequence of points inside a disc of radius 1.
    
    x = path[0]
    y = path[1]
    
    aligned_path = [ conformal_automorphism(x,y,p) for p in path[0:-1] ] + [[0,1]]
    #have to skip the last step to avoid infinities.



#################

def dist(v, w):
    return np.linalg.norm(np.array(v) - np.array(w))


def plot(fairness_vector):
    plt.plot(fairness_vector)
    plt.show()

##################




def integral_disc(r):
    # Takes in r (should be an integer, and finds the integral points in the disc of radius r + .5, then divides by r + .5 to renormalize.
    grid = nx.grid_graph([4 * r, 4 * r])
    relabels = {}
    dual_relabels = {}
    for v in grid.nodes():
        grid.nodes[v]["coord"] = (v[0] - 2*r, v[1] - 2*r)
        relabels[v] = grid.nodes[v]["coord"]
    grid = nx.relabel_nodes(grid, relabels)
    intersection_nodes = [v for v in grid.nodes() if np.linalg.norm(grid.nodes[v]["coord"]) < r + .5]
    disc_graph = nx.subgraph(grid, intersection_nodes)

    #now normalize:
        
    for v in grid.nodes():
        grid.nodes[v]["pos"] = np.array(grid.nodes[v]["coord"]) / (r + .5)

    dual = Facefinder.planar_dual(disc_graph, True)
    disc_graph.graph["dual"] = dual
    return disc_graph







###For Gerrychain:
    
    
def slow_reversible_propose_bi(partition):
    """Proposes a random boundary flip from the partition in a reversible fasion
    by selecting uniformly from the (node, flip) pairs.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    #b_nodes = {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
    #           }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})

    fnode = random.choice(list(partition["b_nodes"]))

    return partition.flip({fnode: -1*partition.assignment[fnode]})

def geom_wait(partition):
    return int(np.random.geometric(len(list(partition["b_nodes"]))/(len(partition.graph.nodes)**(len(partition.parts))-1) ,1))-1


def b_nodes_bi(partition):
    return {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})



def cut_accept(partition):

    bound = 1
    if partition.parent is not None:
        bound = (partition["base"]**(-len(partition["cut_edges"])+len(partition.parent["cut_edges"])))#*(len(boundaries1)/len(boundaries2))



    return random.random() < bound

def new_base(partition):
    return 2.63815853