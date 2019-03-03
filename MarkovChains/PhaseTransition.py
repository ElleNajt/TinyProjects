# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:16:59 2019

@author: Temporary
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import cProfile

def plot(fairness_vector):
    plt.plot(fairness_vector)
    plt.show()



def initialize_graph(size, p):
    grid = nx.grid_graph( [size, size])
    for x in grid.nodes():
        if x[1] < p*size :
            grid.node[x]["vote"] = 0
        else:
            grid.node[x]["vote"] = 1
        if x[1] < size/2:
            grid.node[x]["district"] = 0
        else:
            grid.node[x]["district"] = 1
    return grid

def check_connected(grid):
    district_zero = []
    district_one = []
    for x in grid.nodes():
        if grid.node[x]["district"] == 0:
            district_zero.append(x)
        else:
            district_one.append(x)
    if np.abs(len(district_one) - len(district_zero)) > 2:
        return False
    dist_zero_graph = nx.subgraph(grid, district_zero)
    dist_one_graph = nx.subgraph(grid, district_one)
    if not nx.is_connected(dist_one_graph):
        return False
    if not nx.is_connected(dist_zero_graph):
        return False
    return True

def step(grid):
    x = random.choice(list(grid.nodes()))
    old = grid.node[x]["district"]
    grid.node[x]["district"] = 1 - old

    while not check_connected(grid):
        grid.node[x]["district"] = old
        x = random.choice(list(grid.nodes()))
        old = grid.node[x]["district"]
        grid.node[x]["district"] = 1 - old

    return [grid, x, old]

def cut_size(grid):
    district_zero = []
    district_one = []
    for x in grid.nodes():
        if grid.node[x]["district"] == 0:
            district_zero.append(x)
        else:
            district_one.append(x)


    return len(list(nx.edge_boundary(grid, district_zero, district_one)))


def vote(grid):
    district_zero = []
    district_one = []
    for x in grid.nodes():
        if grid.node[x]["district"] == 0:
            district_zero.append(x)
        else:
            district_one.append(x)
    seats = []
    tally = 0
    for x in district_zero:
        if grid.node[x]["vote"] == 1:
            tally += 1
        else:
            tally += -1
    seats.append(np.sign(tally))
    tally = 0
    for x in district_one:
        if grid.node[x]["vote"] == 1:
            tally += 1
        else:
            tally += -1
    seats.append(np.sign(tally))

    return seats

def viz_vote(graph):
    for x in graph.nodes():
        graph.node[x]["pos"] = x
    values = [graph.node[x]["vote"] for x in graph.nodes()]
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 10, width = .5, cmap=plt.get_cmap('jet'), node_color=values)

def viz_district(graph):
    for x in graph.nodes():
        graph.node[x]["pos"] = x
    values = [graph.node[x]["district"] for x in graph.nodes()]
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 10, width = .5, cmap=plt.get_cmap('jet'), node_color=values)



def fair_vote(vote):
    #This is specifically for a case like 6/10 -- where fairness might mean 50-50 seats.
    if np.mean(vote) == 0:
        return 1
    else:
        return 0

def test():
    grid = initialize_graph(20, .6)

    votes = []
    for i in range(10000):
        grid, x, old = step(grid)
        votes.append(vote(grid))
    fairness_super_critical = [fair_vote(x) for x in votes]
    plot(fairness_super_critical)
    viz_district(grid)


def test_2():
    parameter = .3
    grid = initialize_graph(20,.6)
    #viz_district(grid)
    votes = []
    undids = 0
    for i in range(10000):

        old_cut = cut_size(grid)
        grid, x, old = step(grid)
        new_cut = cut_size(grid)
        successful_sample = False
        while successful_sample == False:
            if new_cut > old_cut:
                p = random.uniform(0, 1)
                cut_off = parameter ** ( new_cut - old_cut)
                if p > cut_off:
                    undids += 1
                    grid.node[x]["district"] = old
                    grid, x, old = step(grid)
                    new_cut = cut_size(grid)
                else:
                    successful_sample = True
            else:
                successful_sample = True
        votes.append(vote(grid))
    fairness_sub_critical = [fair_vote(x) for x in votes]
    #viz_district(grid)

def profile():
    cProfile.run('test_2()')

def make_samples(graph_size, proportion, parameter, num_samples):
    votes = []
    undids = 0
    grid = initialize_graph(graph_size, proportion)
    got_samples = 0
    while got_samples < num_samples:
        old_cut = cut_size(grid)
        grid, x, old = step(grid)
        new_cut = cut_size(grid)
        if new_cut > old_cut:
            p = random.uniform(0, 1)
            cut_off = parameter ** ( new_cut - old_cut)
            if p > cut_off:
                undids += 1
                grid.node[x]["district"] = old
            else:
                got_samples += 1
        else:
            got_samples += 1
        if (got_samples % 10000) == 0:
            print(got_samples)
        votes.append(vote(grid))
    print(undids)
    fairness_vector = [fair_vote(x) for x in votes]

    return [votes, fairness_vector, grid]

def test_around_critical():
    critical_value =0.379

    graph_size = 40
    proportion = .6
    parameter = .35
    num_samples = 1000000


    votes, fairness_sub_critical, sub_critical_grid = \
        make_samples(graph_size, proportion, parameter, num_samples)



    print(np.mean(fairness_sub_critical))

    parameter = .45

    votes, fairness_super_critical, super_critical_grid = \
        make_samples(graph_size, proportion, parameter, num_samples)

    print(np.mean(fairness_super_critical[50000:]))


    plot(fairness_super_critical)
    plot(fairness_sub_critical)