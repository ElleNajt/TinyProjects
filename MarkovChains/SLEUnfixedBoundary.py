
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import Facefinder
from RecursiveGadget import check_simple_cycle, convert_to_edges, add
from SLEExperiments import integral_disc
from Facefinder import planar_dual, draw_with_location, restricted_planar_dual

D = integral_disc(10)
for v in D.nodes():
    D.node[v]["pos"] = D.node[v]["coord"]
dual = planar_dual(D)

'''
We need to get a basis for the faces of dual.
'''
graph = D

double_dual = restricted_planar_dual(restricted_planar_dual(graph))
basis = [convert_to_edges(dual, x) for x in double_dual.nodes()]

#Now add the external face:




def enumerate(dual):
    G = nx.DiGraph()
    for x in dual.nodes():
        G.add_node(x)
    for e in dual.edges():
        G.add_edge(e[0], e[1])
        G.add_edge(e[1], e[0])

    cycle_list = list(nx.simple_cycles(G ))
    #You need the Markov chain! That's what we are trying to analyze


def MC_simple_cycles(graph, steps):
    '''



    :param graph:
    :return:


    '''
    graph = Facefinder.compute_rotation_system(graph)
    graph = Facefinder.compute_face_data(graph)

    dual_R = Facefinder.restricted_planar_dual(graph)

    basis = [convert_to_edges(graph, x) for x in dual_R.nodes()]
    # basis = cycle_basis(graph)
    history = []
    set_basis = []
    for b in basis:
        set_basis.append(frozenset([frozenset(x) for x in b]))
    basis = set_basis
    for i in range(steps):
        x = current
        neighbors = [add(x, b) for b in basis if check_simple_cycle(graph, add(x, b))]
        choice = random.choice(neighbors)
        history.append(choice)


    return history