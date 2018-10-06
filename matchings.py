# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:47:53 2018

@author: Temporary
"""

'''Matchings Markov chain


Reference: Jerrum's Notes, Chapter 5'''

import networkx as nx
import numpy as np
import random
import copy

class matching:
    def __init__(self, graph, edges, weight):
        self.ambient_graph = graph
        self.matching_graph = nx.Graph()
        self.matching_graph.add_nodes_from(sum([ [e[0], e[1]] for e in edges], []))
        self.matching_graph.add_edges_from(edges)
        self.weight = weight
    '''
    
    Span can be accessed via (self.matching_graph.nodes())
    '''
            
    def add_edge(self, edge):
        self.matching_graph.add_edge(edge[0], edge[1])     
    def delete_edge(self, edge):
        self.matching_graph.remove_edge(edge[0], edge[1])
        self.matching_graph.remove_nodes_from([edge[0], edge[1]])
    def delete_edge_by_vertex(self, vertex):
        #Deletes the edge attached to this vertex:
        self.matching_graph.remove_node(vertex)
        self.matching_graph.add_node(vertex)
        
    def step_along_edge(self, edge):
        '''
        If e = {u,v} 
        --If u and v are not in span(M), then add e to matching
        --If e (or e reverse) are in matching , delete it
        --If u is uncovered and v is covered by some e', then delete e' and add e
        --If v is uncovered ...
        Else don't modifying matching
        
        (The other case is if u and v are in the span...)'''
        
        if edge in self.matching_graph.edges():
            self.delete_edge(edge)
            return
        if edge[0] not in self.matching_graph.nodes():
            if edge[1] not in self.matching_graph.nodes():
                self.add_edge(edge)
                return
            if edge[1] in self.matching_graph.nodes():
                self.delete_edge_by_vertex(edge[1])
                self.add_edge(edge)
                return
                
        if edge[1] not in self.matching_graph.nodes():
            if edge[0] in self.matching_graph.nodes():
                self.delete_edge_by_vertex(edge[0])
                self.add_edge(edge)
                return
        return
    
    
    def step(self):
        '''This takes a step using the Metropolis filter'''
        edge = random.choice(list(self.ambient_graph.edges()))
        current_matching = set(self.matching_graph.edges())
        current_weight = self.weight**( len(current_matching))
        self.step_along_edge(edge)
        if self.weight != 1:
            #This metropolis filter is only relevent in the case when self.weight is not 1.
            proposed_matching = set(self.matching_graph.edges())
            proposed_weight = self.weight**( len(proposed_matching))
            c = random.uniform(0,1)
            threshold = min(1, proposed_weight / current_weight)
            #print(threshold, len(new_matching.edges()) - len(current_matching.edges()))
            
            if threshold > c:
                #Then undo the step
                
                #If deleted:
                if edge in current_matching:
                    self.add_edge(edge)
                    return
                if edge in proposed_matching:
                    #If we added the edge -- delete it
                    self.delete_edge(edge)
                    #We might have removed an edge to add edge...
                    removed_edge = current_matching - proposed_matching
                    if len(removed_edge) > 0:
                        self.add_edge(list(removed_edge)[0])

            
    def display(self):
        '''Draw ambient graph with matchings overlayed...'''
        names = {}
        for g in self.ambient_graph.nodes:
            names[g] = g
        nx.draw(self.ambient_graph, pos=names, node_size = 0, node_color= 'black', edge_color = 'blue', width = 8)
        nx.draw(self.matching_graph, pos=names, node_size = 0,  node_color= 'black', edge_color = 'r', width = 8)
        


def display_sum(M1,M2):
    names = {}
    for g in M1.ambient_graph.nodes:
        names[g] = g
    nx.draw(M1.ambient_graph, pos=names, node_size = 0, node_color= 'black', edge_color = 'black', width = 1)
    nx.draw(M1.matching_graph, pos=names, node_size = 0,  node_color= 'black', edge_color = 'red', width = 8)
    nx.draw(M2.matching_graph, pos=names, node_size = 0,  node_color= 'black', edge_color = 'blue', width = 8)
    print("note that this doesn't handle overlapping edgs correctly...)")


G = nx.grid_graph([100,100])
M = matching(G,[((2,2),(2,3))],.01)
lengths = []
for i in range(50000):
    if i == 1000:
        N = copy.deepcopy(M)
    M.step()
    lengths.append(len(M.matching_graph.edges()))
#M.display()      
print(max(lengths))

display_sum(N,M)


def sym_dif(I,F):
    
def canonical_path(I,F)
    #constructs and animates the canonical path from matching I to F...
    nodes = list(I.ambient_graph.nodes())
    ordering= {}
    i = 0
    for node in nodes:
        ordering[node] = i
        i += 1
    I_edges = set(I.matching_graph.edges())
    F_edges = set(F.matching_graph.edges())
    difference = I_edges.symmetric_difference(F_edges)
    difference = I_edges^F_edges
    fixed_ordering = list( I.ambient_graph.nodes())
    difference_graph = nx.Graph()
    difference_graph.add_edges_from(list(difference))
    components = list(nx.connected_components(difference_graph))
    fr_components = [ tuple(sorted(list(x), key = lambda x : ordering[x])) for x in components]
    tuple_ordering = {}
    for x in fr_components:
        minval = min([ ordering[t] for t in x])
        tuple_ordering[x] = minval
    components_sorted = sorted(fr_components, key = lambda x : tuple_ordering[x])
    '''We processes the components in this induced orderging...
    
    
    return 
