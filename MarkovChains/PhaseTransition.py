# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:16:59 2019

@author: Lorenzo
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import cProfile
from SLEExperiments import integral_disc
from mpl_toolkits.mplot3d import Axes3D

def plot(fairness_vector):
    plt.plot(fairness_vector)
    plt.show()



def initialize_graph(size, p):
    grid = nx.grid_graph( [size, size])
    for x in grid.nodes():
        grid.node[x]["zeros"] = 0
        grid.node[x]["ones"] = 0
        if x[1] < p*size :
            grid.node[x]["vote"] = 0
        else:
            grid.node[x]["vote"] = 1
        if x[1] < size/2:
            grid.node[x]["district"] = 0
        else:
            grid.node[x]["district"] = 1
    return grid

def middle_box(size, p = .2, q = .2 + np.sqrt(.4)):
    grid = nx.grid_graph( [size, size])
    for x in grid.nodes():
        grid.node[x]["zeros"] = 0
        grid.node[x]["ones"] = 0
        if (p * size <= x[1] < q*size) and ((p * size <= x[0] < q*size) ):
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
    if len(district_one) == 0:
        return True
    if len(district_zero) == 0:
        return True
    #if np.abs(len(district_one) - len(district_zero)) > 2:
    #    return False
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
        grid.node[x]["zeros"] += 1
        if grid.node[x]["vote"] == 1:
            tally += 1
        else:
            tally += -1
    seats.append(np.sign(tally))
    tally = 0
    for x in district_one:
        grid.node[x]["ones"] += 1
        if grid.node[x]["vote"] == 1:
            tally += 1
        else:
            tally += -1
    seats.append(np.sign(tally))

    return seats

def viz_vote(graph):
    for x in graph.nodes():
        graph.node[x]["pos"] = x
    values = [graph.node[x]["vote"] + 3 for x in graph.nodes()]
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 10, width = .5, cmap=plt.get_cmap('Set2'), node_color=values)

def viz_district(graph):
    for x in graph.nodes():
        graph.node[x]["pos"] = x
    values = [graph.node[x]["district"] for x in graph.nodes()]
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 10, width = .5, cmap=plt.get_cmap('jet'), node_color=values)


def viz_soft_district(graph):
    for x in graph.nodes():
        graph.node[x]["pos"] = x
    values = [graph.node[x]["ones"] - graph.node[x]["zeros"] for x in graph.nodes()]
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 10, width = .5, cmap=plt.get_cmap('jet'), node_color=values)


def viz_soft_district_one(graph):
    for x in graph.nodes():
        graph.node[x]["pos"] = x
    values = [graph.node[x]["zeros"] for x in graph.nodes()]
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 10, width = .5, cmap=plt.get_cmap('binary'), node_color=values)





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
    slopes = []
    undids = 0
    #grid = initialize_graph(graph_size, proportion)
    grid =  middle_box(20)
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
        slopes.append(slope(grid))
    print(undids)
    fairness_vector = [fair_vote(x) for x in votes]

    return [votes, fairness_vector, grid, slopes]

def slope(grid):

    #This will compute the average slope, after tilting $i$ to $i + 1$... this is imperfect, since I'd like
    #this average measurement to be linearly equivariant

    district_zero = []
    district_one = []
    for x in grid.nodes():
        if grid.node[x]["district"] == 0:
            district_zero.append(x)
        else:
            district_one.append(x)

    cuts = list(nx.edge_boundary(grid, district_zero, district_one))
    horizontal = 0
    vertical = 0
    for e in cuts:
        if e[0][0] == e[1][0]:
            vertical += e[0][1] - e[1][1]
        else:
            horizontal += e[0][0] - e[1][0]
    slope = horizontal * np.array( [ 1,0]) + vertical * np.array([0,1])
    #slope = slope / np.linalg.norm(slope) it's not good to get rid of the norm -- we miss information
    return slope

def homology_class(torus):
    #Here's a natural statistic:

    #On the torus, the partitions will correspond to either a: Simple Cycles
    #b: Pairs of non-simple cycles int he same homology class
    #Probably moving between these is slow... is it possible? Yes, because the graph si 2 connected.
    #This case is DEFINATELY SLOW.

    return False


def map_up(grid):

    #Since a key bottle neck seems to be regarding the slopes, I wonder if this is also a problem for mixing as well...
    # All we suspect at this point is that FOR FIXED ENDPOINTS, the mixing at critical temperature appears to be rapid,

    return False






def slopes_plot(slopes):


    plt.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Prepare arrays x, y, z
    z = np.linspace(-2, 2, len(slopes))
    x = [vect[0] for vect in slopes]
    y = [vect[1] for vect in slopes]

    ax.plot(x, y, z)


    plt.show()

def test_slopes():

    slope_list = []
    grid_list = []
    for i in range(5):
        graph_size = 10
        proportion = .6
        parameter = 1
        num_samples = 500000

        votes, fairness, grid, slopes = \
            make_samples(graph_size, proportion, parameter, num_samples)
        slope_list.append(slopes)
        grid_list.append(grid)

    print( [np.mean(x, 0) for x in slope_list])
    print( [ np.var(x,0) for x in slope_list])

def test_around_critical():
    critical_value =0.379

    graph_size = 30
    proportion = .6
    parameter = .35
    num_samples = 2000000


    sub_critical_votes, fairness_sub_critical, sub_critical_grid, slopes = \
        make_samples(graph_size, proportion, parameter, num_samples)


    #plot(fairness_sub_critical)
    viz_soft_district(sub_critical_grid)



    print(np.mean(fairness_sub_critical))

    parameter = 0.379

    super_critical_votes, fairness_super_critical, super_critical_grid, slopes = \
        make_samples(graph_size, proportion, parameter, num_samples)

    print(np.mean(fairness_super_critical))


    plot(fairness_super_critical)

    viz_vote(super_critical_grid)
    viz_soft_district(super_critical_grid)

    plot(fairness_sub_critical)

