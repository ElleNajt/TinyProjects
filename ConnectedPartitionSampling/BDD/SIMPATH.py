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



### For reference, the correct values:
  https://arxiv.org/pdf/cond-mat/0506341.pdf  
1 2
2 12
3 184
4 8512
5 1262816
6 575780564
7 789360053252
8 3266598486981642
9 41044208702632496804

"""
import time
import networkx as nx
import copy
from matplotlib import pyplot as plt
import random
class BDD_node:
    
    def __init__(self, layer, graph, order = 0):
        self.virtual_degrees = {x : 0 for x in graph.nodes()}
        self.virtual_components = { x : {y : False for y in graph.nodes()} for x in graph.nodes()}
        self.layer = layer
        self.order = order # for plotting
        self.graph = graph
        self.arc = {} # stores the two arcs out
        
        self.current_subgraph = [] ## Just  used for debugging to store the current 
        ## set of edges. Only meaningful when we don't identify identical nodes.

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
    for x in graph.nodes():
        root.virtual_components[x][x] = True
    
    BDD = nx.DiGraph()
    BDD.graph["layers"] = m
    BDD.graph["indexing"] = {}
    BDD.graph["layer_widths"] = {}
    BDD.add_node(root)
    BDD.nodes[root]["display_data"] = 'R'
    BDD.nodes[root]["order"] = 0
    BDD.nodes[root]["layer"] = -1
    BDD.graph["indexing"][(-1, 0)] = root
    BDD.graph["layer_widths"][-1] = 1
    for i in [0,1]:
        BDD.add_node(i)
        BDD.nodes[i]["display_data"] = i
        BDD.nodes[i]["order"] = i
        BDD.nodes[i]["layer"] = m-1
        BDD.graph["indexing"][(m-1,i)] = i
    BDD.graph["layer_widths"][m-1] = 2
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
    for layer in range(m): 
        layer_ref = layer + 1 # just to comport with the reference
        order = 0
        for current_node in N[layer]:
            for arc_type in [0,1]: # choice of whether or not to include the edge
                node_new = make_new_node(s,t,current_node, edge_list, frontiers, layer_ref, arc_type) # returns a new node or a 0/1-terminal
                if not ( node_new == 1) and not ( node_new == 0):
                    found_duplicate = False
                    for node_other in N[layer+1]:
                        if identical(node_new, node_other, frontiers[layer_ref], graph):
                            node_new = node_other
                            found_duplicate = True
                            BDD.add_edge(current_node, node_new)
                    if found_duplicate == False:
                        N[layer+1].add( node_new) # add node to ith layer
                        BDD.add_node(node_new) # add the new node to BDD
                        
                        BDD.nodes[node_new]["display_data"] = BDD.nodes[current_node]["display_data"]+ str(arc_type)
                        BDD.nodes[node_new]["order"] = order
                        order += 1
                        BDD.nodes[node_new]["layer"] = layer
                        BDD.graph["indexing"][ ( layer, order - 1)] = node_new
                        BDD.add_edge(current_node, node_new)
                if node_new == 1 or node_new == 0:
                    BDD.add_edge(current_node, node_new)
                current_node.arc[arc_type] = node_new #set the x pointer of node to node_new
                #print(BDD.nodes[current_node]["display_data"])
                #for x in current_node.virtual_components.keys():
                #    print(x, current_node.virtual_components[x])
                #print('   ')
        if layer != m - 1:
            BDD.graph["layer_widths"][layer] = order
    return BDD

def make_new_node(s, t, current_node, edge_list, frontiers, layer_ref, arc_type):
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
    edge = edge_list[layer_ref - 1]
    
    v = edge[0]
    w = edge[1]
    #print(edge, frontiers[layer + 1])
    if arc_type == 1:
        # this is the case of adding edge i
        if current_node.virtual_components[v][w] == True:
            # There already was a path from v to w, so we would create a cycle
            # by including 
            #print("REJECTED")
            return 0
    current_node_copy = copy.deepcopy(current_node)
    update_node_info(current_node_copy, edge_list, frontiers, layer_ref, arc_type)
    
    for u in [v,w]:
        if (u == s or u == t) and current_node_copy.virtual_degrees[u] > 1:
            #print(0)
            return 0
        if current_node_copy.virtual_degrees[u] > 2:
            #print(0)
            return 0
        # two more termination conditions
    for u in [v,w]:
        if u not in frontiers[layer_ref]:
            # u is never going to be touched again, so has to be in the 
            # final form -- needs to be +2, since the frontier is always
            # one behind the layer
            if (u in [s,t]) and current_node_copy.virtual_degrees[u] != 1:
                return 0
            if (u not in [s,t]):
                #print(u, current_node_copy.virtual_degrees[u])
                if current_node_copy.virtual_degrees[u] not in [0,2]:
                    #print("returned")
                    return 0
            
            # now since u leaves the frontier, we can do some memory management
            # jsut by removing it from the dictionary, python will handle rest
    if layer_ref - 1== len(edge_list) - 1:
        # this was the last edge, and we found no contradictions
        return 1
    return current_node_copy

def update_node_info(node, edge_list, frontiers, layer_ref, arc_type):
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
    edge = edge_list[layer_ref - 1]
    v = edge[0]
    w = edge[1]
    for u in [v,w]:
        if u not in frontiers[layer_ref - 1]:
            #this means that u has entered the frontier for the first time
            #check indexing
            node.virtual_degrees[u] = 0 # this might be redundant
            node.virtual_components[u][u] = True
    if arc_type == 1:
        
        node.current_subgraph.append(edge)
        for u in [v,w]:
            node.virtual_degrees[u] += 1
            
        merge_set = dict({})
        
        v_component = node.virtual_components[v]
        w_component = node.virtual_components[w]            
        #print("before", v, w)
        #print([t for t in v_component.keys() if v_component[t] == 1])
        #print([t for t in w_component.keys() if w_component[t] == 1])
        for x in node.virtual_components.keys():
            merge_set[x] =  v_component[x] or w_component[x]
            
        for t in node.virtual_components.keys():
            node.virtual_components[v][t] = merge_set[t]
            node.virtual_components[w][t] = merge_set[t]
        #print("after)")
        #print([t for t in v_component.keys() if node.virtual_components[v][t] == 1])
        #print([t for t in w_component.keys() if node.virtual_components[w][t] == 1])
        ## Symmetrize
        for t in node.virtual_components.keys():
            for u in node.virtual_components.keys():
                if node.virtual_components[u][t] == True:
                    node.virtual_components[t][u] = True
    
    
        ## Now take the transitive closure!
        
        merged_component = [ t  for t in node.virtual_components.keys() if node.virtual_components[v][t] == True]
        
        for x in merged_component:
            for y in merged_component:
                node.virtual_components[x][y] = True
        
    return 

def identical(node_1, node_2, frontier, graph):
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
    Returns True if R(node_1) == R(node_2), where
R(n) is the set of edges sets corresponding to paths from n to 1. 

    """
    # frontier= graph.nodes() # Just for debugging
    for vertex in frontier:

        if node_1.virtual_degrees[vertex] != node_2.virtual_degrees[vertex]:
            return False
    for vertex_1 in frontier:
        for vertex_2 in frontier:

            if node_1.virtual_components[vertex_1][vertex_2] != node_2.virtual_components[vertex_1][vertex_2]:
                return False
    return True

def debugging_connectivity(BDD, graph):
    
    '''
    
    In the unsplit node setting, each node has the current subgraph in it.
    
    '''
    
    for node in BDD.nodes():
        if node != 0 and node != 1:
            edges = node.current_subgraph
            subgraph = nx.edge_subgraph(graph, edges).copy()  
            for y in graph.nodes():
                subgraph.add_node(y) ## Because otherwise doesn't have all the vertices
            components = nx.connected_components(subgraph)
            disagreement = False
            for x in graph.nodes():
                for y in graph.nodes():
                    connected_in_edges = nx.has_path(subgraph, x,y)
                    connected_in_data_1 = node.virtual_components[x][y]
                    connected_in_data_2 = node.virtual_components[y][x]
                    
                    if len (set ( [connected_in_edges, connected_in_data_1, connected_in_data_2])) == 2:
                        ## We have disagreement!
                        disagreement = True
                        if connected_in_edges and not connected_in_data_1:
                            print("data missing a connection")
                        if connected_in_data_1 and not connected_in_edges:
                            print("too many connections")
                        
                        
            # So it seems like what could be happening here
            # is that connectivity is not propagating correctly... but then there's also more edges 
            if disagreement == True:
                print("found disagreement")
                print(edges)
                
                virtual_edges = []
                for edge in graph.edges():
                    if node.virtual_components[edge[0]][edge[1]] == True:
                        virtual_edges.append(edge)
                print(virtual_edges)
                edge_color = {}
                for x in graph.edges():
                    edge_color[x] = 1
                for y in path:
                    edge_color[y] = 0
                
                edge_colors = [edge_color[edge] for edge in graph.edges()]
                coords = {}
    
                for x in graph.nodes():
                    coords[x] = x
                nx.draw(graph,pos = coords, edge_color= edge_colors, with_labels = True, widht = 4)
                plt.savefig(str(time.time()) + ".png")
def count_accepting_paths(BDD):
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
    BDD.nodes[0]["count"] = 0
    BDD.nodes[1]["count"] = 1
    m = BDD.graph["layers"]
    for i in range(m-2, -2,-1):
        for j in range(BDD.graph["layer_widths"][i]):
            current_node = BDD.graph["indexing"][(i,j)]
            left_child = current_node.arc[0]
            right_child = current_node.arc[1]
            BDD.nodes[current_node]["count"] = BDD.nodes[left_child]["count"]  + BDD.nodes[right_child]["count"] 
        
    return BDD.nodes[BDD.graph["indexing"][(-1, 0)]]["count"]


def enumerate_accepting_paths(BDD):
    
    ## Goal of this is mostly to debug the path counting
    
    BDD.nodes[0]["set"] = set()
    BDD.nodes[1]["set"] = set(['T'])
    m = BDD.graph["layers"]
    for i in range(m-2, -2,-1):
        for j in range(BDD.graph["layer_widths"][i]):
            current_node = BDD.graph["indexing"][(i,j)]
            left_child = current_node.arc[0]
            right_child = current_node.arc[1]
            BDD.nodes[current_node]["set"] = set()
            for c in [0,1]:
                child = [left_child, right_child][c]
                for x in BDD.nodes[child]["set"]:
                    BDD.nodes[current_node]["set"].add( str(c) + x)
            #print(BDD.nodes[current_node]["set"])
    
    return BDD.nodes[BDD.graph["indexing"][(-1, 0)]]["set"]

for scale in range(1,10):
    print("size, " , scale + 1)
    left_dim = 1+ scale
    right_dim = 1 + scale
    graph = nx.grid_graph([left_dim,right_dim])
    s = (0,0)
    t = (right_dim-1,left_dim-1)
    
    #graph = nx.barbell_graph(3,3)
    #s = 0
    #t = 8
    
    edge_list = list( graph.edges())
    
    #random.shuffle(edge_list)
    # A random order is *much* worse!
    
    m = len(edge_list)
    
    
    
    simpath = simple_paths(graph, edge_list, s,t)
    
    display_labels = { x : simpath.nodes[x]["display_data"] for x in simpath.nodes()}
    
    display_coordinates = { x : (simpath.nodes[x]["order"]*1000 ,m - simpath.nodes[x]["layer"]) for x in simpath.nodes()}
    
    display_coordinates[0] = ( .3,m - simpath.nodes[0]["layer"] )
    display_coordinates[1] = ( .6,m - simpath.nodes[0]["layer"] )
    
    print("size of BDD", len(simpath))
    #print(simpath.graph["layer_widths"])
    
    #101111001011T'
    
    
    # debugging_connectivity(simpath, graph)
    
    print("number of paths", count_accepting_paths(simpath))           

'''
paths = list(enumerate_accepting_paths(simpath))




paths_as_edgelists = []
bad_path = []

coords = {}

for x in graph.nodes():
    coords[x] = x
for x in paths:
    path_edges = []
    for i in range(len(edge_list)):
        if x[i] == '1':
            path_edges.append(edge_list[i])
    paths_as_edgelists.append(path_edges)

for path in paths_as_edgelists:
    subgraph = nx.edge_subgraph(graph, path)
    if not (nx.is_tree(subgraph)):
        if len(list(nx.connected_components(subgraph))) > 0 :
            bad_path_binary = paths[paths_as_edgelists.index(path)]
            bad_path = path
            #print(bad_path)
                
                
        edge_color = {}
        for x in graph.edges():
            edge_color[x] = 1
        for y in path:
            edge_color[y] = 0
        
        edge_colors = [edge_color[edge] for edge in graph.edges()]
        
        nx.draw(graph,pos = coords, edge_color= edge_colors, with_labels = True, widht = 4)
        plt.savefig(str(time.time()) + ".png")
    plt.close()
    
    
#bad_path = paths_as_edgelists[5]
'''
'''
edge_color = {}
for x in graph.edges():
    edge_color[x] = 0
for y in bad_path:
    edge_color[y] = 1

edge_colors = [edge_color[edge] for edge in edge_list]

nx.draw(graph, edge_color= edge_colors, with_labels = True)
    
    

simpath.remove_node(0)

arc_types = {}
for edge in simpath.edges():
    if edge[0].arc[0] == edge[1]:
        arc_types[edge] = 'r'
    if edge[0].arc[1] == edge[1]:
        arc_types[edge] = 'b'
    if edge[0].arc[0] == edge[1] and edge[0].arc[1] == edge[1]:
        arc_types[edge] = 'g'

#
#for node in simpath.nodes():
#    if simpath.nodes[node]["display_data"] == 'R0':
#        strange_node = node
#(strange_node, strange_node.arc[1]) in simpath.edges()       
        
        
arc_colors = [arc_types[edge] for edge in simpath.edges()]


nx.draw(simpath, pos = display_coordinates, edge_color = arc_colors, labels = display_labels, with_labels = True, node_size = 100)

'''