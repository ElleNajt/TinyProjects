import os
import random
import json
import geopandas as gpd
import functools
import datetime
import matplotlib
from facefinder import *
from graph_tools import *
import time
import graph_tools

import matplotlib.pyplot as plt
import numpy as np
import csv
from networkx.readwrite import json_graph
import math
import seaborn as sns
from functools import partial
import networkx as nx
import numpy as np
import copy

import seannas_code

from gerrychain.tree import bipartition_tree as bpt
from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population, UpperBound)
from gerrychain.proposals import propose_random_flip, propose_chunk_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain import GeographicPartition
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap
from gerrychain.tree import recursive_tree_part, bipartition_tree_random, PopulatedGraph, contract_leaves_until_balanced_or_none, find_balanced_edge_cuts

# functions below are tools needed for metamandering experiment
def graph_from_url_processing(link):
    r = requests.get(url=link)
    data = json.loads(r.content)
    g = json_graph.adjacency_graph(data)
    graph = Graph(g)
    graph.issue_warnings()
    for node in graph.nodes():
        graph.nodes[node]["pos"] = [graph.nodes[node]['C_X'], graph.nodes[node]['C_Y'] ]
    deg_one_nodes = []
    for v in graph:
        if graph.degree(v) == 1:
            deg_one_nodes.append(v)
    for node in deg_one_nodes:
        graph.remove_node(node)
    return graph

def build_trivial_partition(graph):
    assignment = {}
    for y in graph.nodes():
        assignment[y] = 1
    first_node = list(graph.nodes())[0]
    assignment[first_node] = -1
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    partition = Partition(graph, assignment=assignment, updaters=updaters)
    return partition

def build_balanced_partition(graph, pop_col, pop_target, epsilon):
    
    block = my_mst_bipartition_tree_random(graph, pop_col, pop_target, epsilon)
    assignment = {}
    for y in graph.nodes():
        if y in block:
            assignment[y] = 1
        else:
            assignment[y] = -1
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    partition = Partition(graph, assignment=assignment, updaters=updaters)
    return partition


def build_balanced_k_partition(graph, k, pop_col, pop_target, epsilon):
    
    assignment = recursive_tree_part(graph, k, pop_target, pop_col, epsilon)
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    partition = Partition(graph, assignment=assignment, updaters=updaters)
    return partition
    
    

def build_partition_meta(graph, mean):
    assignment = {}
    for y in graph.nodes():
        if graph.nodes[y]['C_Y'] < mean:
            assignment[y] = -1
        else:
            assignment[y] = 1
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    partition = Partition(graph, assignment=assignment, updaters=updaters)
    print("cut edges are", partition["cut_edges"])
    return partition

def assign_special_faces(graph, k):
    special_faces = []
    for node in graph.nodes():
        if graph.nodes[node]['distance'] >= k:
            special_faces.append(node)
    return special_faces


def remove_undirected_edge(graph, v, u):
    if (v,u) in graph.edges():
        graph.remove_edge(v,u)
        return 
    
    if (u,v) in graph.edges():
        graph.remove_edge(u,v)
        return 
    
    #print("nodes ", v, ",", u, " not connected in graph")


def face_sierpinski_mesh(graph, special_faces):
    #parameters: 
    #graph: graph object that edges will be added to
    #special_faces: list of faces that we want to add node/edges to
    #k: integer depth parameter for depth of face refinement
    max_label = max(list(graph.nodes()))
    for face in special_faces:
        graph.add_node(face)
        neighbor_list = []
        locations = []
        connections = []
        location = np.array([0,0]).astype("float64")
        for v in face:
            neighbor_list.append(v)
            location += np.array(graph.nodes[v]["pos"]).astype("float64")
        graph.nodes[face]["pos"] = location / len(face)
        for w in face:
            locations.append(graph.nodes[w]["pos"] - graph.nodes[face]["pos"])
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)
        for v in range(0,len(neighbor_list)):
            next_index = (v+1) % len(neighbor_list)
            distance = np.array(graph.nodes[neighbor_list[v]]["pos"]) + np.array(graph.nodes[neighbor_list[next_index]]["pos"])
            distance = distance * .5
            label = max_label + 1
            max_label += 1
            graph.add_node(label)
            graph.nodes[label]['pos'] = distance
            remove_undirected_edge(graph, neighbor_list[v], neighbor_list[next_index])
            graph.add_edge(neighbor_list[v],label)
            graph.add_edge(label,neighbor_list[next_index])
            connections.append(label)
        for v in range(0,len(connections)):
            if v+1 < len(connections):
                graph.add_edge(connections[v],connections[v+1])
            else:
                graph.add_edge(connections[v],connections[0])
        graph.remove_node(face)
    return graph

def cut_accept(partition):
    bound = 1
    if partition.parent is not None:
        bound = (partition["base"] ** (-len(partition["cut_edges"]) + len(
            partition.parent["cut_edges"])))  # *(len(boundaries1)/len(boundaries2))
    return random.random() < bound

def step_num(partition):
                parent = partition.parent

                if not parent:
                    return 0

                return parent["step_num"] + 1
            
            
def get_spanning_tree_mst(graph):
    for edge in graph.edges:
        graph.edges[edge]["weight"] = random.random()

    spanning_tree = nx.tree.maximum_spanning_tree(
        graph, algorithm="kruskal", weight="weight"
    )
    return spanning_tree


def my_mst_bipartition_tree_random(
    graph,
    pop_col,
    pop_target,
    epsilon,
    node_repeats=1,
    spanning_tree=None,
    choice=random.choice):
    populations = {node: graph.nodes[node][pop_col] for node in graph}

    possible_cuts = []
    if spanning_tree is None:
        spanning_tree = get_spanning_tree_mst(graph)

    while len(possible_cuts) == 0:
        spanning_tree = get_spanning_tree_mst(graph)
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = find_balanced_edge_cuts(h, choice=choice)

    return choice(possible_cuts).subset


def always_true(proposal):
    return True
           
# Experiement setup

def smooth_node(graph, v):
    #print(v)
    neighbors = list(graph.neighbors(v))
    graph.remove_node(v)
    try:
        graph.add_edge(neighbors[0], neighbors[1])
    except:
        print(neighbors)
        
        
    ##TODO : This can create parallel edges, which can result in degree 2 nodes 
    #turning into degree 1 nodes after this smoothing. Those are ok, sicne they are just deleted
    #But really we have to keep looping the clean up.
    
    return graph

def preprocessing(which_map):
    maps = [ "https://people.csail.mit.edu/ddeford/COUSUB/COUSUB_13.json", "https://people.csail.mit.edu/ddeford/COUSUB/COUSUB_55.json", "https://people.csail.mit.edu/ddeford/COUNTY/COUNTY_13.json"]

    
    link = maps[which_map]
    graph = graph_from_url_processing(link)
    
    
    #Have to remove bad nodes in order for the duality thing to work properly
    

    cleanup = True
    while cleanup:
        print("clean up phase")
        print(len(graph))
        deg_one_nodes = []
        for v in graph.nodes():
            if graph.degree(v) == 1:
                deg_one_nodes.append(v)
        graph.remove_nodes_from(deg_one_nodes)
        
        deg_2_nodes = []
        for v in graph.nodes():
            if graph.degree(v) == 2:
                deg_2_nodes.append(v)
    
        for v in deg_2_nodes:
            graph = smooth_node(graph, v)    
        
        bad_nodes = []
        for v in graph.nodes():
            if graph.degree(v) == 1 or graph.degree(v) == 2:
                bad_nodes.append(v)
        if len(bad_nodes) > 0:
            cleanup = True
        else:
            cleanup = False
        
    print(len(graph))
    print("making dual")
    dual = restricted_planar_dual(graph)
    print("made dual")
    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'))
    plt.savefig("./plots/UnderlyingGraph.png", format='png')
    plt.close()
    
    
    for node in graph.nodes():
        graph.nodes[node]["pos"] = [graph.nodes[node]["C_X"], graph.nodes[node]["C_Y"]]
        graph.nodes[node]["population"] = graph.nodes[node]["POP10"]


    return graph, dual


def produce_sample(graph, k, tag, sample_size = 500):
    #Samples k partitions of the graph, stores the cut edges and records them graphically
    #Also stores vote histograms, and returns most extreme partitions.
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['cut_times'] = 0
    
        for n in graph.nodes():
            #graph.nodes[n]["population"] = 1 #graph.nodes[n]["POP10"] #This is something gerrychain will refer to for checking population balance
            graph.nodes[n]["last_flipped"] = 0
            graph.nodes[n]["num_flips"] = 0
    
    #sierp_partition = build_balanced_partition(g_sierpinsky, "population", ideal_population, .01)
    
    
    
    ideal_population= sum( graph.nodes[x]["population"] for x in graph.nodes())/k
    initial_partition = build_balanced_k_partition(graph, list(range(k)), "population", ideal_population, .1)
    #viz(g_sierpinsky, set([]), sierp_partition.parts)
    pop1 = .1
    
    
    popbound = within_percent_of_ideal_population(initial_partition, pop1)
    #ideal_population = sum(sierp_partition["population"].values()) / len(sierp_partition)
    print(ideal_population)
    
    tree_proposal = partial(recom,pop_col="population",pop_target=ideal_population,epsilon= 1 ,node_repeats=1)
    steps = sample_size
    
    
    chaintype = "tree"
    
    if chaintype == "tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=my_mst_bipartition_tree_random)
    
    if chaintype == "uniform_tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=my_uu_bipartition_tree_random)
    
    
    
    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=always_true, initial_state=initial_partition, total_steps=steps)
    
    
    z = 0
    num_cuts_list = []
    
    
    seats_won_table = []
    best_left = np.inf
    best_right = -np.inf
    
    for part in exp_chain:
    
    #for i in range(steps):
    #    part = build_balanced_partition(g_sierpinsky, "population", ideal_population, .05)
    
        seats_won = 0
        z += 1
        
        if z % 100 == 0:
            print("step ", z)
    
        for edge in part["cut_edges"]:
            graph[edge[0]][edge[1]]["cut_times"] += 1
    
        for i in range(k):
            rural_pop = 0
            urban_pop = 0
            for n in graph.nodes():
                if part.assignment[n] == i:
                    rural_pop += graph.nodes[n]["RVAP"]
                    urban_pop += graph.nodes[n]["UVAP"]
            total_seats = int(rural_pop > urban_pop)
            seats_won += total_seats
        seats_won_table.append(seats_won)
        if seats_won < best_left:
            best_left = seats_won
            left_mander = copy.deepcopy(part.parts)
        if seats_won > best_right:
            best_right = seats_won
            right_mander = copy.deepcopy(part.parts)
        #print("finished round"
    
    print("max", best_right, "min:", best_left)
    
    edge_colors = [graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()]
    
    pos=nx.get_node_attributes(graph, 'pos')
    
    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=1,
                        edge_color=edge_colors, node_shape='s',
                        cmap='magma', width=3)
    plt.savefig("./plots/edges" + tag + ".png")
    plt.close()
    
    plt.figure()
    plt.hist(seats_won_table, bins = 10)
    
    name = "./plots/seats_histogram" + tag +".png"
    plt.savefig(name)
    plt.close()    

    return left_mander, right_mander

def metamander_around_partition(graph, dual, target_partition, tag):
    updaters = {'population': Tally('population'),
                        'cut_edges': cut_edges,
                        'step_num': step_num,
                        }
    
    assignment = {}
    for x in graph.nodes():
        color = 0
        for block in target_partition.keys():
            if x in target_partition[block]:
                assignment[x] = color
            color += 1
    
    target_partition = Partition(graph, assignment, updaters = updaters)
    plt.figure()
    
    viz(graph, set([]), target_partition.parts)
    plt.savefig("./plots/target_map" + tag + ".png", format = 'png')
    plt.close()
    
    print("made partition")
    crosses = compute_cross_edge(graph, target_partition)
    
    k = len(target_partition.parts)
    
    dual_crosses = []
    for edge in dual.edges:
        if dual.edges[edge]["original_name"] in crosses:
            dual_crosses.append(edge)
            
    print("making dual distances")
    dual = distance_from_partition(dual, dual_crosses)
    print('finished making dual distances')
    special_faces = assign_special_faces(dual,2)
    print('finished assigning special faces')
    g_sierpinsky = face_sierpinski_mesh(graph, special_faces)
    print("made metamander")
    
    for node in g_sierpinsky:
        g_sierpinsky.nodes[node]['C_X'] = g_sierpinsky.nodes[node]['pos'][0]
        g_sierpinsky.nodes[node]['C_Y'] = g_sierpinsky.nodes[node]['pos'][1]
        if 'population' not in g_sierpinsky.nodes[node]:
            g_sierpinsky.nodes[node]['population'] = 0
        if 'RVAP' not in g_sierpinsky.nodes[node]:
            g_sierpinsky.nodes[node]['RVAP'] = 0
        if 'UVAP' not in g_sierpinsky.nodes[node]:
            g_sierpinsky.nodes[node]['UVAP'] = 0
        ##Need to add the voting data
    total_pop = sum( [ g_sierpinsky.nodes[node]['population'] for node in g_sierpinsky])
    
    #sierp_partition = build_trivial_partition(g_sierpinsky)
    
    plt.figure()
    nx.draw(g_sierpinsky, pos=nx.get_node_attributes(g_sierpinsky, 'pos'), node_size = 1, width = 1, cmap=plt.get_cmap('jet'))
    plt.savefig("./plots/sierpinsky_mesh.png", format='png')
    plt.close()
    
    left_mander, right_mander = produce_sample(g_sierpinsky, k , tag)
   

def test_and_find_left_right_manders(graph):

    k = 6
    tag = "state_map" + str(which_map) + "trial_num" + str(trial)
    ##Number of Partitions Goes Here
    

    left_mander, right_mander = produce_sample(graph, k, "finding")

    
    return left_mander, right_mander
 
    
def metamander_experiment():
    which_map = 0

    graph, dual = preprocessing(which_map)
    
    left_mander, right_mander = test_and_find_left_right_manders(graph)
    
    hold_graph = copy.deepcopy(graph)
    hold_dual = copy.deepcopy(dual)

    
    metamander_around_partition(graph, dual, left_mander, tag + "LEFTMANDER")
    
    graph = hold_graph
    dual = hold_dual
    ##This should work but doesn't seem to have to call preprocessing again... 
    #probably because of dual
    graph, dual = preprocessing(which_map)
    metamander_around_partition(graph, dual, right_mander, tag + "RIGHTMANDER")
    
metamander_experiment()
