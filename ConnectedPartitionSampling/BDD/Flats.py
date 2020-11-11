# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:51:26 2020

@author: lnajt

An implementation of Kawahara et al. Frontier search algorithm specialized to flats.

Adapted from Kawahara et al.

Notational changes

Kawahara:
    
    i -> layer
    x -> arc_type
    
    
    
A note on the data structures:
    
    node.virtual_components[v][w] is a dictionary that tells you if is connected. (Needs to be maintained as symmetric and transitive closed.)
    to w -- it is only maintained and guaranteed correct for the particular 
    Frontier set that is relevant! The rest is just junk, but allocated in
    memory anyway.

    node.virtual_discomponent[v][w] -- maintains whether v cannot be connected to w. Symmetric, but not transitive. If v~w (connected) and w ~_a u then v ~_a u ... in other words, is propagated by the connected components 



###Known Results from naive backtracking:
https://oeis.org/A145835 gives "1, 12, 1434, 1691690, 19719299768, 2271230282824746, 2584855762327078145444, 29068227444022728740767607050, 3230042572278849047360048508956727420, 3546545075986984198328715750838554116235343894"
    2x2: 12

    3x3: 1,434
    
    2x3 : 74

    5x3: 538,150

    4x4: 1,691,690 (about an hour of computation)
    Lower bound from spanning tree: 32768
    Compare to the upper bound from edge subsets: 16,777,216

    (Weird -- this suggests that there's a decent probability of just picking a connected partition??)

    ---

    For 3D graphs:

    2x2x2: 958 ( < 1 second, .058)
    3x2x2: 81,224 ( about 7 seconds)
    3x3x2 : 975,00,024   [ 16397.374621391296 seconds - 4.5 hours]


"""
import time
import networkx as nx
import copy
from matplotlib import pyplot as plt
import random

import pickle 
class BDD_node:
    
    def __init__(self, layer, graph, order = 0):
        self.virtual_components = { x : {y : False for y in graph.nodes()} for x in graph.nodes()}
        for x in graph.nodes():
            self.virtual_components[x][x] = True
        self.virtual_discomponent = { x : {y : False for y in graph.nodes()} for x in graph.nodes()}
        self.layer = layer
        self.order = order # for plotting
        self.graph = graph
        self.arc = {} # stores the two arcs out
        
        self.current_subgraph = [] ## Just  used for debugging to store the current 
        ## set of edges. Only meaningful when we don't identify identical nodes.

def flats(graph, edge_list):
    """
    Parameters
    ----------
    graph : graph
        DESCRIPTION.
    edge_list : list
        DESCRIPTION.
    Returns The BDD
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
                node_new = make_new_node(current_node, edge_list, frontiers, layer_ref, arc_type) # returns a new node or a 0/1-terminal
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

        if layer != m - 1:
            BDD.graph["layer_widths"][layer] = order
    return BDD

def make_new_node(current_node, edge_list, frontiers, layer_ref, arc_type):
    """
    Parameters
    ----------
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
    
    if arc_type == 0:
        if current_node.virtual_components[v][w] == True:
            # v and w are connected, so must have
            # the edge between them.
            return 0

    if arc_type == 1:
        if current_node.virtual_discomponent[v][w] == True:
            # v and w are not connected, so adding
            # an edge between them would be a contradiction
            return 0
    
    current_node_copy = copy.deepcopy(current_node)
    update_node_info(current_node_copy, edge_list, frontiers, layer_ref, arc_type)
    

    if contradictory(current_node_copy):
        print("found contradiction -- should never happen")
        return 0
    
    if layer_ref - 1== len(edge_list) - 1:
    # this was the last edge, and we found no contradictions
        return 1
    
    return current_node_copy

def contradictory(node):
    for s in node.virtual_components.keys():
        for t in node.virtual_components.keys():
            if node.virtual_components[s][t] == True and node.virtual_discomponent[s][t] == True:
                return True
    return False

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

    if arc_type == 0:
        
        
        for m,n in [(v,w), (w,v)]:
        
            component_m = [ t  for t in node.virtual_components.keys() if node.virtual_components[m][t] == True]
            component_n = [ t  for t in node.virtual_components.keys() if node.virtual_components[n][t] == True]
            for x in component_m:
                for y in component_n:
                    node.virtual_discomponent[x][y] = True
                    node.virtual_discomponent[y][x] = True
            
        
    if arc_type == 1:
        node.current_subgraph.append(edge)
        
        
        


        # Extending Connectivity:
        
        merge_set = dict({})
        v_component = node.virtual_components[v]
        w_component = node.virtual_components[w]            
        for x in node.virtual_components.keys():
            merge_set[x] =  v_component[x] or w_component[x]
            
        for t in node.virtual_components.keys():
            node.virtual_components[v][t] = merge_set[t]
            node.virtual_components[w][t] = merge_set[t]

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
        
        
        
        ## Now use the extended connectivity to update the anti-connectedness.
        
        for t in node.virtual_components.keys():
            anti_connected = False
            for x in merged_component:
                if node.virtual_discomponent[t][x] == True or node.virtual_discomponent[x][t] == True:
                    # need to make sure it is symmetric
                    anti_connected = True
            if anti_connected == True:
                for x in merged_component:
                    node.virtual_discomponent[t][x] = True
                    node.virtual_discomponent[x][t] = True
        
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

    for vertex_1 in frontier:
        for vertex_2 in frontier:

            if node_1.virtual_components[vertex_1][vertex_2] != node_2.virtual_components[vertex_1][vertex_2]:
                return False
            if node_1.virtual_discomponent[vertex_1][vertex_2] != node_2.virtual_discomponent[vertex_1][vertex_2]:
                return False
    return True

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

for scale in range(1,5):
    left_dim = scale
    right_dim = scale
    
    dimensions = [left_dim, right_dim]
    print("dimensions: ", dimensions)
    graph = nx.grid_graph(dimensions)

    edge_list = list( graph.edges())
    
    # random.shuffle(edge_list)
    # A random order is *much* worse!
    
    m = len(edge_list)
    
    
    
    BDD = flats(graph, edge_list)
    
    display_labels = { x : BDD.nodes[x]["display_data"] for x in BDD.nodes()}
    
    display_coordinates = { x : (BDD.nodes[x]["order"]*1000 ,m - BDD.nodes[x]["layer"]) for x in BDD.nodes()}
    
    display_coordinates[0] = ( .3,m - BDD.nodes[0]["layer"] )
    display_coordinates[1] = ( .6,m - BDD.nodes[0]["layer"] )
    
    print("size of BDD", len(BDD))
    
    #101111001011T'
    

    print("number of flats", count_accepting_paths(BDD))    

    BDD_name = str(dimensions) + ".p"       

    #pickle.dump( BDD, open( BDD_name, "wb"))

    '''
    paths = list(enumerate_accepting_paths(BDD))
    
    paths_as_edgelists = []

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
    
                
        edge_color = {}
        for x in graph.edges():
            edge_color[x] = 1
            if x in path or (x[1], x[0]) in path:
                edge_color[x] = 0
        
        edge_colors = [edge_color[edge] for edge in graph.edges()]
        
        nx.draw(graph,pos = coords, edge_color= edge_colors, with_labels = True, width = 4)
        plt.savefig(str(time.time()) + ".png")
        plt.close()
    
    '''

'''
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


    

    
paths_as_sets = [ set(x) for x in paths_as_edgelists]
cleaned_parts = [ set(x) for x in cleaned_partitions]
missing_parts = []
for x in cleaned_parts:
    found = False
    for y in paths_as_sets:
        passed = False
        if len(x) == len(y):
            passed = True
            for edge in x:
                if (edge not in y) and ((edge[1], edge[0]) not in y):
                    passed = False
        if passed == True:
            found = True
    if found == False:
        missing_parts.append(x)
        
print(len(missing_parts))


missing_parts = []
for x in paths_as_sets:
    found = False
    for y in cleaned_parts:
        passed = False
        if len(x) == len(y):
            passed = True
            for edge in x:
                if edge not in y and (edge[1], edge[0]) not in y:
                    passed = False
        if passed == True:
            found = True
    if found == False:
        missing_parts.append(x)
        
print(len(missing_parts))



for path in missing_parts:
    subgraph = nx.edge_subgraph(graph, path)

            
    edge_color = {}
    for x in graph.edges():
        edge_color[x] = 1
    for y in path:
        edge_color[y] = 0
    
    edge_colors = [edge_color[edge] for edge in graph.edges()]
    
    nx.draw(graph,pos = coords, edge_color= edge_colors, with_labels = True, width = 4)
    plt.savefig(str(time.time()) + ".png")
    plt.close()


'''