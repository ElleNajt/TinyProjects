# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:34:31 2019

@author: Lorenzo
"""

import networkx as nx
import copy
import random
import numpy as np
import math
import mpmath
import time
import matplotlib.pyplot as plt
def partition_extendable(graph, in_set):
    coloring = graph.graph["coloring"]
    for e in in_set:
        if coloring[e[0]] == coloring[e[1]]:
            return False
    return True

def update_coloring(graph, edge):
    #Here we add edge to the set of non-cut edges, meaning that the two endpoints are now definately in the same connected compoennt. WE update the coloring to reflect that.
    coloring = graph.graph["coloring"]
    a = coloring[edge[0]]
    b = coloring[edge[1]]
    if a == b:
        return graph
        #If already in the same component, do nothing.
    else:
        #Update the component coloring by the smallest label.
        #TODO: I think there's a way to speed this up, by redefining "b" to be "a" in some way (or vica versa)
        if a < b:
            for x in graph.nodes():
                if coloring[x] == b:
                    coloring[x] = a
        else:
            for x in graph.nodes():
                if coloring[x] == a:
                    coloring[x] = b
    return graph

def back_tracking_recursion(input_graph, layer, in_set, out_set, new_in = set(), new_out = set()):
    #graph = copy.deepcopy(input_graph)
    #This was a hack to manage color resetting. Now works better with the old_coloring idea.

    graph = input_graph
    old_coloring = copy.deepcopy(graph.graph["coloring"])
    if new_out != set():
        graph = update_coloring(graph, new_out)

    extendable = partition_extendable(graph, in_set)
    #print(layer, in_set, out_set, graph.graph["coloring"])

    #print(extendable)

    if extendable == False:
        graph.graph["coloring"] = old_coloring
        return [False]

    if layer == len(graph.edges()):
        graph.graph["coloring"] = old_coloring
        return [in_set]

    processed_edge= graph.graph["ordered_edges"][layer]
    layer += 1
    #A corresponding skip ahead step would go here.
    left_tree = back_tracking_recursion(graph, layer, in_set + [processed_edge], out_set, processed_edge, set())
    right_tree = back_tracking_recursion(graph, layer, in_set, out_set + [processed_edge], set(),processed_edge)

    graph.graph["coloring"] = old_coloring
    #TODO: Make sure these coloring updates are organized correctly.

    return left_tree + right_tree


def backtracking(graph):
    layer = 0
    graph.graph["ordered_edges"] = list(graph.edges())

    #print(graph.graph["ordered_edges"])
    graph.graph["coloring"] = {x : x for x in graph.nodes()}
    #Each node will start in its own block, when we declare an edge to be in the outset, will will update the colors of the blocks contianing those two nodes to be the same. THis is done by having the smallest color win.

    in_set = []
    out_set = []
    return back_tracking_recursion(graph, layer, in_set, out_set)

def number_partitions_backtracking(input_graph):
    graph =  nx.convert_node_labels_to_integers(input_graph)

    list_of_partitions = backtracking(graph)
    #pruned_list = [x for x in list_of_partitions if x != False]
    counter = 0
    for m in list_of_partitions:
        if m != False:
            counter += 1
    return counter


#input_graph = nx.grid_graph([3,3])
#number_partitions_backtracking(input_graph)


'''


###Results:

    2x2: 12

    3x3: 1,434

    5x3: 538,150

    4x4: 1,691,690 (about an hour of computation)
    Lower bound from spanning tree: 32768
    Compare to the upper bound from edge subsets: 16,777,216

    (Weird -- this suggests that there's a decent probability of just picking a connected partition??)


 '''




def rejection_sample(graph):
     #the above experiments suggest that, at least for small grid graphs, taking a random edge subset,  there is a decent chance of that being the cut-set of a connected partition.
    J = []
    I = []
    for e in graph.edges():
        c = random.uniform(0,1)
        if c > .5 :
            J.append(e)
        else:
            I.append(e)

    graph.graph["coloring"] = {x : x for x in graph.nodes()}

    for e in J:
        graph = update_coloring(graph, e)

    coloring = graph.graph["coloring"]
    for e in I:
        if coloring[e[0]] == coloring[e[1]]:
            return False
    return [J, coloring]

def viz(graph, edge_set, coloring, name):


    convert = {}
    coloring_convert = {}
    node_list = list(graph.nodes())
    for i in range(len(node_list)):
        convert[node_list[i]] = i
    for i in range(len(node_list)):
        coloring_convert[node_list[i]] = convert[coloring[node_list[i]]]


    for x in graph.nodes():
        graph.nodes[x]["pos"] = [x[0], x[1]]
    values = [1 - int(x in edge_set) for x in graph.edges()]
    node_values = [convert[coloring[x]] for x in graph.nodes()]
    f = plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_color = node_values, edge_color = values, labels = coloring_convert, width = 4, node_size= 65, font_size = 7)
    f.savefig(name + str(int(time.time())) + ".png")

def test_rejection_sample(graph, goal = 1):

    samples = []

    sample_colors = []

    number = 0

    while number < goal:
        new = rejection_sample(graph)

        if new != False:
            number += 1
            #print("got one")
            samples.append(new)
            sample_colors.append(graph.graph["coloring"])

    #k = 2
    #edge_set = samples[k]
    #coloring = sample_colors[k]
    #viz(graph, edge_set, coloring)

    print("Got ", number, " samples")

    return samples

'''

This works pretty well for 6x6

For 7x7 got 11 samples in 100000 tries
For 8x8 got 2 samples in 100000 tries
For 9x9 got 1 sample in 1,000,000 tries

Next: You can use the pruning from above -- just take a random path in the backtracking tree, and restart the path if "extendable" ever declares it not extendable.

'''



def restarting_rejection_sample(graph):

    layer = 0
    graph.graph["ordered_edges"] = list(graph.edges())

    #print(graph.graph["ordered_edges"])
    graph.graph["coloring"] = {x : x for x in graph.nodes()}
    #Each node will start in its own block, when we declare an edge to be in the outset, will will update the colors of the blocks contianing those two nodes to be the same. THis is done by having the smallest color win.

    in_set = []
    out_set = []
    return restarting_rejection_recursion(graph, layer, in_set, out_set)


def restarting_rejection_recursion(input_graph, layer, in_set, out_set, new_in = set(), new_out = set()):
    #graph = copy.deepcopy(input_graph)
    #This was a hack to manage color resetting. Now works better with the old_coloring idea.

    graph = input_graph
    #old_coloring = copy.deepcopy(graph.graph["coloring"])
    if new_out != set():
        graph = update_coloring(graph, new_out)

    extendable = partition_extendable(graph, in_set)
    #print(layer, in_set, out_set, graph.graph["coloring"])

    #print(extendable)

    if extendable == False:
        #graph.graph["coloring"] = old_coloring
        return [False]

    if layer == len(graph.edges()):
        #graph.graph["coloring"] = old_coloring
        return [in_set]

    processed_edge= graph.graph["ordered_edges"][layer]
    layer += 1
    #A corresponding skip ahead step would go here.
    coin = random.uniform(0,1)
    if coin < 1/2:
        return back_tracking_recursion(graph, layer, in_set + [processed_edge], out_set, processed_edge, set())
    else:
        return back_tracking_recursion(graph, layer, in_set, out_set + [processed_edge], set(),processed_edge)

    #graph.graph["coloring"] = old_coloring
    #TODO: Make sure these coloring updates are organized correctly.

def test_restarting_rejection(graph):

    samples = []
    #graph = nx.grid_graph([6,6])

    for i in range(1000):
        samples.append(restarting_rejection_sample(graph))


    pruned_samples = [x for x in samples if x != False]

    print("Got ", len(pruned_samples), " samples")


'''


I'm not sure that you really win anything from this -- of course the tree is not very deep, so if anything this just means that each sample happens slightly faster. On the other hand, checking this condition at each step is costy.

It would be better to learn something from this process that guides the distribution over edges...


If you want speed ups -- probably doing this with F_2 linear algebra will be faster.

a) Write the incidence matrix
b) Pick a random column.
c) Scan the remaining columns to see if anything was dependent on it.

What is the distribution get by taking the span? It's the distribution over the G[J], which is biased by the number of connected spanning substructures. Since the blocks are (???) always small, this is a pretty small number. Maybe importance sample is reasonable here.

'''

def count_spanning_substructures(matroid):
    #Counts spanning sets.
    #In the graph context -- this is spanning + connected subgraphs. Annoying clash of terminology here --


    return 0

def importance_sampler(graph):

    return 0


'''

Here's a natural Markov chain to try:

    Walk on {0,1}^E
    Each subset J has a # of contradictions, which is the number of edges in J^c that fail the coloring test
    Give score lambda^{#contradictions}.
    Output the samples we see with zero contradictions.

    For lambda = 1, is rapid mixing, but chance of seeing a sample is low.
    For lambda = epsilon, is slowly mixing (???), but stationary is concentrated on the zero contradiction set.

'''

def coloring_split(graph, non_cut_edges, edge):

    #This removes edge from the coloring -- if that was the bridge between the component of that coloring, it assigns an unused color to one of the components.

    coloring = graph.graph["coloring"]
    a = coloring[edge[0]]
    b = coloring[edge[1]]
    #We know a == b...
    if a != b:
        print("something went wrong!")
        print(non_cut_edges)
        print(edge)
        print(coloring)
        viz(graph, non_cut_edges, graph.graph["coloring"])
        return None

    color_component = [x for x in graph.nodes() if coloring[x] == a]

    color_subgraph = nx.Graph(nx.subgraph(graph, color_component))
    #Passing to nx.Graph is necessary to make a mutable subgraph

    color_subgraph_edge_list = list(color_subgraph.edges())
    for f in color_subgraph_edge_list:
        #We need this loop so that we don't accidentally throw in all of the induced edges of the color component
        if f not in non_cut_edges:
            if (f[1], f[0]) not in non_cut_edges:
                color_subgraph.remove_edge(f[0], f[1])

    #color_subgraph.remove_edge(edge[0], edge[1])

    components = list(nx.connected_components(color_subgraph))

    if len(components) == 1:
        #This was the case that e was not a bridge
        return graph

    #Now, in the case that e was a bridge, we need to reassign the colors. First we find a color not used by the other components.

    possible_colors = set(graph.nodes())
    for used_color in coloring.values():
        if used_color in possible_colors:
            possible_colors.remove(used_color)

    unused_color = list(possible_colors)[0]


    #We let component[0] keep its color, and reassign the colors in component[1].
    for y in components[1]:
        #print(y)
        coloring[y] = unused_color

    graph.graph["coloring"] = coloring
    return graph


def contradictions(graph, cut_edges):

    coloring = graph.graph["coloring"]
    contradictions = 0
    for e in cut_edges:
        if coloring[e[0]] == coloring[e[1]]:
            contradictions += 1
    return contradictions

def step(graph, cut_edges, non_cut_edges, temperature):

    coin = random.uniform(0,1)


    if coin < 1/2:
        return non_cut_edges, cut_edges

    old_non_cut_edges = copy.deepcopy(non_cut_edges)
    old_cut_edges = copy.deepcopy(cut_edges)
    old_coloring = copy.deepcopy(graph.graph["coloring"])

    current_contradictions = contradictions(graph, cut_edges)

    e = random.choice(graph.graph["ordered_edges"])

    if e in non_cut_edges:
        #print(e, " is now in cut set!")
        non_cut_edges.remove(e)
        cut_edges.add(e)
        graph = coloring_split(graph, non_cut_edges, e)
        #This is the update coloring that potentially reassigns the colors, because when making e cut some new components might emerge...
        #Be sure to use if else here, otherwise it just undoes itself :-D
    else:
        #print(e, "is now in merge set!")
        non_cut_edges.add(e)
        cut_edges.remove(e)
        graph = update_coloring(graph, e)
        #This is the update coloring that merges the two colors, based on moving e into non-cut.

    #viz(graph, non_cut_edges, graph.graph["coloring"])

    new_contradictions = contradictions(graph, cut_edges)
    #print(new_contradictions)

    if new_contradictions <= current_contradictions:
        return non_cut_edges, cut_edges

    coin = random.uniform(0,1)

    if temperature**(new_contradictions - current_contradictions) <= coin:
        #print("reset, at threshold:", temperature**(new_contradictions - current_contradictions) )
        non_cut_edges = old_non_cut_edges
        cut_edges = old_cut_edges
        graph.graph["coloring"] = old_coloring
        return non_cut_edges, cut_edges
    else:
        return non_cut_edges, cut_edges


def run_markov_chain(graph, steps = 100, temperature = .5):
    #Return a sample as [Cutedges, non_cutedges, coloring]

    #temperature = .8
    #steps = 1000
    graph.graph["ordered_edges"] = list(graph.edges())
    graph.graph["coloring"] = {x : x for x in graph.nodes()}

    cut_edges = set( graph.graph["ordered_edges"])
    non_cut_edges = set([])

    samples = []

    for i in range(steps):
        non_cut_edges, cut_edges = step(graph, cut_edges, non_cut_edges, temperature)
        if contradictions(graph, cut_edges) == 0:
            samples.append([copy.deepcopy(cut_edges),copy.deepcopy(non_cut_edges), copy.deepcopy(graph.graph["coloring"])])

    print(len(samples))

    return samples

def function_to_partition(function):
    #Takes a function, and returns the fiber partition of the ground set

    blocks = []
    list_of_values = list(set(function.values()))
    ground_set = list(function.keys())
    for val in list_of_values:
        block = []
        for x in ground_set:
            if function[x] == val:
                block.append(x)
        blocks.append(block)
    return blocks
def dobinski_random_variable(n):

    '''
    This returns a sample from the distribution on N distributed like P(U = u) = (1 / eB_n) (u^n )/ n!
    By Dobinski's formula this is well defined
    '''

    b = int(mpmath.bell(n))

    c = np.random.uniform(0,1)
    i = 0
    cumulative_probability = 0

    while cumulative_probability < c:
        i += 1
        cumulative_probability += (i**n)/(math.factorial(i)*np.e*b)

    return i
def sample_partition(input_set):
    #returns a uniformly random partition of input_set

    #For the correctness of this algorithm: https://www.sciencedirect.com/science/article/pii/0097316583900092

    num_urns = dobinski_random_variable(len(input_set))
    while num_urns == 0:
        num_urns = np.random.poisson()
    urns = list(range(num_urns))
    function = {}
    for x in input_set:
        color = random.choice(urns)
        function[x]= color

    partition = function_to_partition(function)

    return partition
def sample_partitions(input_set, number):
    partitions = []
    for i in range(number):
        partitions.append(sample_partition(input_set))
    return partitions
def test_complete_graph():
    for size_of_graph in [20,25,30,35,40,45]:

        parameter_values = [i / size_of_graph for i in range(1,5)]
        for parameter in parameter_values:
            print("parameter: ", parameter, "size: ", size_of_graph)
            graph = nx.complete_graph(size_of_graph)

            samples = run_markov_chain(graph, 300000,parameter)
            #samples_as_subgraphs = [nx.edge_subgraph(graph, x[1]) for x in samples]
            #samples_as_partitions = [ (list(nx.connected_components( nx.edge_subgraph(graph, x[1])) )) for x in samples]

            sample_num_components = [len (set(x[2].values())) for x in samples]
            print(np.mean(sample_num_components))

        true_samples = sample_partitions(list(range(size_of_graph)), 7000)
        true_samples_num_components = [len (x) for x in true_samples]
        print("true:",np.mean(true_samples_num_components))
def test_grid_graph():

    size = 15
    graph = nx.grid_graph([size, size])
    MC_steps = 100000
    MC_temperature = .7
    num_true_samples = 0

    samples = run_markov_chain(graph, MC_steps, MC_temperature)
    #Higher temperatures -- the chain mixes faster, but the is less likely to give you a connected partition.
    #So, set the temperature higher if you want to be more confident that any samples you get are uniform.

    #for sample in true_samples:
    #    viz(graph, sample[0], sample[1])
    viz(graph, samples[-10][1], samples[-10][2], "markov_chain")

    #The question: Can we tune temp so that we get both rapid mixing AND polynomial concentration on the zero contradiction assignments??

    #We can falsify rapid mixing using the totally uniform samples that we have.

    non_cut_sets = [len(s[1]) for s in samples]

    sample_num_components = [len (list(
    nx.connected_components( nx.edge_subgraph(graph, x[1])) )) for x in samples]

    true_samples = test_rejection_sample(graph, num_true_samples)

    true_non_cut_sets = [len(x[0]) for x in true_samples]
    true_num_components = [len (list( nx.connected_components( nx.edge_subgraph(graph, x[0])) )) for x in true_samples]

    print(np.mean(non_cut_sets), " vs. ", np.mean(true_non_cut_sets))

    print(len(true_samples))
    for sample in true_samples:
        viz(graph, sample[0], sample[1],"true_uniform")

    print(np.mean(sample_num_components), " vs. ", np.mean(true_num_components))

    ##It seems to pass this test.

#test_grid_graph()
'''
This Markov chain won't mix rapidly in general. We can look for miracles in the structure
of the flats for the grid graph case. Places to look:

1) In the backgtracking tree, we can look for an unusual degree of symmetry.
It might make more sense to look at the torus graph.
2) We can look for lower bounds on the number of flats in this case. Maybe
there are so many of them that
'''


#test_grid_graph()
'''

On a 6x6 with steps= 10000 for both, with temperature = .1

6053
Got  18  samples
22.7127044441  vs.  22.8333333333

With only 1000 MC steps, 26.9242618742  vs.  22.8333333333

----

graph = nx.grid_graph([8,8])
MC_steps = 10000
true_sample_trials = 100000



'''

'''
A stricter test to try?

-- Probably will fail on large complete graphs, because you have to move througe high contradiction arrangements to get a one without a contradiction.


Whatabout number of colors?

--
'''
