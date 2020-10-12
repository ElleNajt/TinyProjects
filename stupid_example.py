# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:28:23 2020

@author: lnajt
"""


import networkx as  nx

G = nx.grid_graph([100,100])

def first_thing():

    table = []
    for n in G.nodes():
        for m in G.nodes():
            table.append([n,m])


def second_thing():
    table_2 = []
    
    for n in G.nodes():
        table_2.append(n)
     #   
#first_thing()

#second_thing()


def function():
    return [1,2]

a,b = function()