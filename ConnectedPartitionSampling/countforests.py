'''
This runs the basis exchange walk to approximate the number of independent sets of size k.


'''

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

def independent_rank_k(graph, subset,k):
    """This takes a graph and a subset of edges, subset, and k, and returns
    True if subset is an independent set of size k, and False otherwise.
    """
    if len(subset) != k:
        return False

    return nx.is_forest(nx.edge_subgraph(graph, subset))

def basis_exchange(graph, independent_set):
    """This takes a single step on the UPDOWN basis exchange graph
    Promised that current_set is independent
    """
    edge = random.choice(list(independent_set.edges()))
    independent_set.remove_edge(edge[0], edge[1])

    other_edges =  list(graph.edges())
    random.shuffle(other_edges)
    for edge in other_edges:
        if edge not in independent_set.edges():
            independent_set.add_edge(edge[0], edge[1])
            if nx.is_forest(independent_set):
                return independent_set
            else:
                independent_set.remove_edge(edge[0], edge[1])


def initialize_independent_set(graph, rank):
    """Creates a rank k independent subset of graph.edges()
    """

    tree = nx.minimum_spanning_tree(graph)
    edges = set(tree.edges())
    n = len(edges)
    while n > rank:
        n += -1
        edge = random.choice(list(edges))
        edges = edges - set([edge])
    independent_set = nx.Graph(nx.edge_subgraph(graph, edges))
    return independent_set

def estimate_num_independent_sets(graph, rank):
    """Uses the basis exchange walk to estimate the number of independent sets
    of size rank. This uses self-reducibility.
    """
    return 0

def closure(graph, independent_set):
    """Returns the flat spanned by the edge_set
    """
    for v in graph.nodes():
        if v not in independent_set.nodes():
            independent_set.add_node(v)
    components = nx.connected_components(independent_set)
    block_subgraphs = []
    for block in components:
        block_subgraphs.append( nx.subgraph(graph, block))
    flat = nx.Graph()
    for block in block_subgraphs:
        flat = nx.union(flat, block)
    return flat

def is_min(graph, independent_set):
    """Determines if independent_set is the cheapest set in its closure
    """
    flat = closure(graph, independent_set)

    ind_weight = 0
    for edge in independent_set.edges():
        ind_weight += graph.graph["weight"][edge]

    min_forest = nx.minimum_spanning_tree(flat, 'weight')

    min_weight = 0
    for edge in min_forest.edges():
        min_weight += graph.graph["weight"][edge]

    if ind_weight == min_weight:
        return True
    return False

def viz(graph, name):
    pos = {x : graph.nodes[x]["pos"] for x in graph.nodes()}
    node_list = list(graph.nodes())
    components = list(nx.connected_components(graph))
    component_dictionary = {}
    num_components = len(components)
    for x in graph.nodes():
        for i in range(num_components):
            if x in components[i]:
                if len(components[i]) == 1:
                    component_dictionary[x] = -1
                else:
                    component_dictionary[x] = i
    colors = [component_dictionary[x] for x in node_list]
    f = plt.figure()
    nx.draw(graph, pos, node_size = 100, width =.5, node_color = colors)
    f.savefig(name + ".png")

def produce_sample(graph, rank, tree_steps = 1000):
    #graph must be initialized with weights already
    independent_set = initialize_independent_set(graph, rank)
    trials = 0
    while True:
        for i in range(tree_steps):
            updated = basis_exchange(graph, independent_set)
            independent_set= updated
        #print(len(independent_set.edges()))
        flat = closure(graph, independent_set)
        if is_min(graph, independent_set):
            return flat
        if trials > 10**10:
            print("timed out")
            return False
        
def initialize_weights(graph):
    weights  = {}
    i = 0
    for edge in graph.edges():
        weights[edge] = 2**i
        weights[ (edge[1], edge[0]) ] = 2**i
        graph.edges[edge]['weight'] = 2**i
        i += 1

    graph.graph["weight"] = weights
    return graph

def test():
    graph_size = 30
    graph = nx.grid_graph([graph_size,graph_size])
    for v in graph.nodes():
        graph.nodes[v]["pos"] = [v[0], v[1]]

    graph = initialize_weights(graph)
    
    rank = 100

    trials = 1000
    tree_steps = 1000
    successes = 0
    for k in range(trials):
        independent_set = initialize_independent_set(graph, rank)
        #print(len(independent_set.edges()))
        for i in range(tree_steps):
            updated = basis_exchange(graph, independent_set)
            independent_set= updated
        #print(len(independent_set.edges()))
        flat = closure(graph, independent_set)
        #viz(flat)
        if is_min(graph, independent_set):
            successes += 1
            name = "flatno" + str(k) + "graphsize" + str(graph_size) + "rank" + str(rank) + "treesteps" + str(tree_steps)
            viz(flat, name)
            print("success")
    print(successes/trials)

#test()
