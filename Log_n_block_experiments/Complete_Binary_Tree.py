# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 02:14:29 2020

@author: lnajt
"""


import networkx as nx
import random
from matplotlib import pyplot as plt
import math
import numpy as np

for degree in [2,3,4]:
    print("degree", degree)
    for depth in range(4,16):
        G = nx.balanced_tree(degree,depth)
        edge_list = list(G.edges())
        max_block_lengths = []
        edge_sizes = []
        for i in range(3000):
            edge_subset = list(filter( lambda x : random.uniform(0,1) < .5, edge_list ))
            subgraph = G.edge_subgraph(edge_subset)
            max_component = max([len(x) for x in (nx.connected_components(subgraph))])
            max_block_lengths.append(max_component)
            
        plt.hist(max_block_lengths)
        print(depth, " ratio: ", max(max_block_lengths)/math.log(len(G),2),  np.mean(max_block_lengths)/math.log(len(G),2))