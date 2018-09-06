# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:14:25 2018

@author: Lorenzo Najt
"""

import networkx as nx
import random
import copy

def nw_most(nodes):
    #returns the nw_most from among the nodes; that is,the element which is largest in the North/South x West/East lexicographic order.
    y_max = max( [node[1] for node in nodes])
    y_maximizers = [ node for node in nodes if node[1] == y_max]
    west_most = y_maximizers[0][0]
    for node in y_maximizers:
        if node[0] < west_most:
            west_most = node[0]
    return (west_most, y_max)

class pentomino_object:
    def __init__(self, torus, root = False):
        self.torus = torus
        self.node_history = []
        self.degree_history = []
        self.stuck = False
        if root == False:
            self.degree_history.append(torus.graph["size"]**2)
            self.initialize_root()
        else:
            self.root = root
        
        self.nodes = {self.root}


    def initialize_root(self):
        n = self.torus.graph["size"]
        root = (random.choice(range(n)), random.choice(range(n)))
        self.root = root
    
    def get_candidates(self):
        '''returns the set of blocks that make pentomino when added, so that the parent of that pentomino is  the current pentomino '''
        
        balls = { frozenset({ (x[0] - 1, x[1]), (x[0] + 1, x[1]), (x[0], x[1]-1), (x[0], x[1] +1)}) for x in self.nodes }
        ball = set().union(*balls)
        boundary = ball.difference(self.nodes)
        candidates = []
        for node in boundary:
            self.nodes.add(node)
            if node == self.get_northwestmost_legal():
                candidates.append(node)
            self.nodes.remove(node)
        return candidates
    
    def get_admissible_nodes(self):
        #returns the nodes in the pentomino that can be removed without disconnecting it
        #Can speed up by looking at boundary
        admissible = []
        for node in self.nodes:
            self.nodes.remove(node)
            n = self.torus.graph["size"]
            block_A = { covering_map(p, n) for p in self.nodes}
            G_A = nx.subgraph(self.torus,block_A)
            if nx.is_connected(G_A):
                admissible.append(node)
            self.nodes.add(node)
        return admissible
    
    def get_northwestmost_legal(self):
        admissible = self.get_admissible_nodes()
        return nw_most(admissible)
    

    
    def valid_candidates(self):
        '''eliminates the blocks that would make the child no longer valid'''
        candidates = self.get_candidates()
        valid_candidates = []
        for candidate in candidates:
            self.nodes.add( candidate)
            n = self.torus.graph["size"]
            block_A = { covering_map(p, n) for p in self.nodes}
            block_B = set(self.torus.nodes).difference(block_A)
            G_B = nx.subgraph(self.torus, block_B)
            if nx.is_connected(G_B):
                valid_candidates.append(candidate)
            self.nodes.remove(candidate)
        self.degree = len(valid_candidates)
        return valid_candidates
    
    def grow(self):
        if self.stuck == True:
            return
        valid_candidates = self.valid_candidates()
        if len(valid_candidates) == 0:
            self.stuck = True
            return
        choice = random.choice(valid_candidates)
        
        self.node_history.append(copy.deepcopy(self.nodes))
        self.degree_history.append(self.degree)
        self.nodes.add(choice)
        
    def likelihood(self):
        product = 1
        for d in self.degree_history:
            product = product * d
        return 1 / product
        

def covering_map(point, n):
    #Sends a point in the lattice to the corresponding point in the n by n discrete torus
    return (point[0] % n, point[1] % n)

def test_validity(pentomino_nodes, G):
    '''
    G: The nxn torus graph
    
    pentomino is in the lattice. We check if the image in the torus graph is a block of a connected partition.
    '''
    n = G.graph["size"] 
    
    block_A = { covering_map(p, n) for p in pentomino_nodes}
    block_B = [p for p in G.nodes() if p not in block_A]
    G_A = nx.subgraph(G,block_A)
    G_B = nx.subgraph(G,block_B)
    
    if not nx.is_connected(G_A):
        return False
    if not nx.is_connected(G_B):
        return False
    return True

def create_torus(n):
    C = nx.cycle_graph(n)
    torus = nx.cartesian_product(C,C)
    torus.graph["size"] = n
    return torus

def generate_tiling(torus, size):
    
    pentomino = pentomino_object(torus)
    i = 1
    while i < size:
        i += 1
        pentomino.grow()
    #Necessary to stop it regardless of whether the pentomino got stuck or not
    return pentomino

def integrate_from_samples(function, samples):
    '''samples is a collection of pentominos'''
    sum = 0
    for pentomino in samples:
        sum += function(pentomino) / pentomino.likelihood()
    return sum / len(samples)

def constant_one(pentomino):
    return 1

def cut(pentomino, power = 1):
    block = { covering_map(p, pentomino.torus.graph["size"]) for p in pentomino.nodes}
    return nx.cut_size(pentomino.torus, block)**power

def make_samples(torus, size, trials = 10):
    samples = []
    for i in range(trials):
        pentomino = generate_tiling(torus, size)
        if pentomino.stuck == False:
            samples.append(pentomino)
    return samples

def integrate(function, torus, size, trials = 10):
    return integrate_from_samples(function, make_samples(torus, size, trials))


def sanity_check(n = 10, size = 5, num_samples = 1000):
    '''This is a good debugger, because this counts the number of tetrominos. The correct answer is 19.
    For size= 5, the correct answer is 63'''
    torus = create_torus(n)
    tests = []
    for i in range(1):
        samples = make_samples(torus, size, num_samples)
        tests.append(integrate_from_samples(constant_one, samples))
    print([test / (n**2) for test in tests])
    
    counter = {}
    for pentomino in samples:
        counter[pentomino.likelihood()] = 0
    for pentomino in samples:
        counter[pentomino.likelihood()] += 1
    counter


def cutsize(n = 10, size = "half", power = 1, num_samples = 10, trials = 10):
    if size == "half":
        size = n**2 / 2
    torus = create_torus(n)
    tests = []
    for i in range(trials):
        samples = make_samples(torus, size, num_samples)
        print( " Succesfully built ", len(samples), "Pentominos")
        tests.append(integrate_from_samples(cut, samples) / integrate_from_samples(constant_one, samples))
    print(tests)
    
cutsize(6)
#m =6 
#pents = make_samples(create_torus(m), (m**2)/2,1)
#[p.stuck for p in pents]
p = generate_tiling(create_torus(6), 18)
