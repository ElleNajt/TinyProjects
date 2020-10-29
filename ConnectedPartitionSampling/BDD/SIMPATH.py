# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:51:26 2020

@author: lnajt

An implementation of the simple path algorithm using BDDs

Adapted from Kawahara et al.

Notational changes

Kawahara:
    
    i -> layer
    x -> arc_type
    
    
    
A note on the data structures:
    
    node.virtual_degree is a dictionary. We keep it defined on all vertices
    but only the allocations on the Frontier matter for all operations
    node.virtual_components[v][w] is a dictionary that tells you if is connected
    to w -- it is only maintained and guaranteed correct for the particular 
    Frontier set that is relevant! The rest is just junk, but allocated in
    memory anyway.

"""

import networkx as nx
import copy

class BDD_node:
    
    def __init__(self, layer, graph):
        self.virtual_degrees = {x : 0 for x in graph.nodes()}
        self.virtual_components = { x : {y : False for y in graph.nodes()} for x in graph.nodes()}
        self.layer = layer
        self.graph = graph
        self.arc = {} # stores the two arcs out

def simple_paths(graph, edge_list,s,t):
    """
    Parameters
    ----------
    graph : TYPE
        DESCRIPTION.
    edge_list : TYPE
        DESCRIPTION.
    s : vertex
        start vertex
    t : vertex
        terminal vertex

    Returns
    -------
    None.

    """
    
    
    root = BDD_node("root", graph)

    m = len(edge_list)
    N = [set( [root])]  # change to layer nodes or something
    for i in range(1, m+1):
        N.append( set() )
    # N is an auxiliary function that will create track of the layers
    
    BDD = nx.DiGraph()
    BDD.add_node(root)
    BDD.nodes[root]["display_data"] = 'R'
    BDD.add_node(0)
    BDD.add_node(1)
    BDD.nodes[0]["display_data"] = 0
    BDD.nodes[1]["display_data"] = 1
    ## Create Frontier Sets
    frontiers = [] 
    # Note F_-1 = \emptyset -- if comparing to Kawahara et al., note that
    # their edge indices start at 1.
    for i in range(m+1):
        left_subgraph = graph.edge_subgraph(edge_list[:i])
        right_subgraph = graph.edge_subgraph(edge_list[i:])
        frontier_set = set ( left_subgraph.nodes() ).intersection( set( right_subgraph.nodes()))
        frontiers.append(frontier_set)
    ##
    for layer in range(m): # i in reference
        for current_node in N[layer]:
            for arc_type in [0,1]: # choice of whether or not to include the edge
                node_new = make_new_node(s,t,current_node, edge_list, frontiers, layer, arc_type) # returns a new node or a 0/1-terminal
                if not ( node_new == 1) and not ( node_new == 0):
                    found_duplicate = False
                    for node_other in N[layer+1]:
                        if identical(node_new, node_other, frontiers[layer+1]):
                            node_new = node_other
                            found_duplicate = True
                    if found_duplicate == False:
                        N[layer+1].add( node_new) # add node to ith layer
                        BDD.add_node(node_new) # add the new node to BDD
                        BDD.nodes[node_new]["display_data"] = BDD.nodes[current_node]["display_data"]+ str(arc_type)
                        BDD.add_edge(current_node, node_new)
                if node_new == 1 or node_new == 0:
                    BDD.add_edge(current_node, node_new)
                current_node.arc[arc_type] = node_new #set the x pointer of node to node_new
    
    return BDD

def make_new_node(s, t, current_node, edge_list, frontiers, layer, arc_type):
    """
    Parameters
    ----------
    s : vertex
        start vertex
    t : vertex
        terminal vertex
    current_node : TYPE
        DESCRIPTION.
    edge_list : TYPE
        DESCRIPTION.
    frontiers : set
        the list of frontier sets
    layer : number
        the current layer
    arc_type : TYPE
        DESCRIPTION.

    Returns
    -------
    This populates the new nodes info, nand does checks to see if it should 
    return 0 or 1 instead.

    """
    edge = edge_list[layer]
    
    v = edge[0]
    w = edge[1]
    if arc_type == 1:
        # this is the case of adding edge i
        if current_node.virtual_components[v][w] == True:
            # There already was a path from v to w, so we would create a cycle
            # by including edge
            return 0
    current_node_copy = copy.deepcopy(current_node)
    update_node_info(current_node_copy, edge_list, frontiers, layer, arc_type)
    
    for u in [v,w]:
        if (u == s or u == t) and current_node_copy.virtual_degrees[u] > 1:
            #print(0)
            return 0
        if (u != s and u != t) and current_node_copy.virtual_degrees[u] > 2:
            #print(0)
            return 0
        # two more termination conditions
    for u in [v,w]:
        if u not in frontiers[layer + 1]:
            # u is never going to be touched again, so has to be in the 
            # final form -- needs to be +2, since the frontier is always
            # one behind the layer
            if (u == s or u == t) and current_node_copy.virtual_degrees[u] != 1:
                return 0
            if (u != s and u != t):
                if current_node_copy.virtual_degrees[u] not in [0,2]:
                    return 0
            
            # now since u leaves the frontier, we can do some memory management
            # does python handle this well?            
            #memory_manage(current_node_copy, u)
    if layer == len(edge_list) - 1:
        # this was the last edge, and we found no contradictions
        return 1
    return current_node_copy

def update_node_info(node, edge_list, frontiers, layer, arc_type):
    """
    

    Parameters
    ----------
    node : TYPE
        DESCRIPTION.
    edge : TYPE
        DESCRIPTION.
    arc_type : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    edge = edge_list[layer]
    v = edge[0]
    w = edge[1]
    for u in [v,w]:
        if u not in frontiers[layer]:
            #this means that u has entered the frontier for the first time
            #check indexing
            node.virtual_degrees[u] = 0 # this might be redundant
            node.virtual_components[u][u] = 1
    if arc_type == 1:
        for u in [v,w]:
            node.virtual_degrees[u] += 1
            merge_set = {}
            
            v_component = node.virtual_components[v]
            w_component = node.virtual_components[w]            
            
            for x in node.virtual_components.keys():
                merge_set[x] = max( [v_component[x], w_component[x]] )
            node.virtual_components[v] = merge_set
            node.virtual_components[w] = merge_set
    
    return 

def identical(node_1, node_2, frontier):
    """
   
    Parameters
    ----------
    node_1 : BDD node
        DESCRIPTION.
    node_2 : BDD node
        DESCRIPTION.
    frontier : TYPE
        DESCRIPTION.

    Returns a boolean
    -------
    Returns True if R(node_1) = R(node_2), where
R(n) is the set of edges sets corresponding to paths from n to 1. 

    """
    
    for vertex in frontier:
        if node_1.virtual_degrees[vertex] != node_2.virtual_degrees[vertex]:
            return False
    for vertex_1, vertex_2 in frontier:
        if node_1.virtual_components[vertex_1][vertex_2] != node_2.virtual_components[vertex_1][vertex_2]:
            return False
    return True



graph = nx.grid_graph([2,2])
s = (0,0)
t = (1,1)
edge_list = list( graph.edges())

simpath = simple_paths(graph, edge_list, s,t)

display_labels = { x : simpath.nodes[x]["display_data"] for x in simpath.nodes()}


nx.draw_planar(simpath, labels = display_labels, with_labels = True)
def count_accepting_paths(BDD,root):
    """
    Parameters
    ----------
    BDD : TYPE
        DESCRIPTION.
    root : TYPE
        DESCRIPTION.

    Returns
    -------
    Number of paths from root to 1.

    """
    