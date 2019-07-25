
import os

#os.chdir('/home/lorenzonajt/Documents/GerrychainSensitivity/PA_VTD')
os.chdir('/home/lorenzonajt/Documents/GITHUB/TinyProjects/OutlierFugacitySensitivity')
from gerrychain import Graph, GeographicPartition, Partition, Election
from gerrychain.updaters import Tally, cut_edges
import geopandas as gpd
import numpy as np
import random
import copy

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
from gerrychain.metrics import polsby_popper
from gerrychain import constraints

import matplotlib.pyplot as plt


import pandas


def analyze_dem_seats(chain):

    d_percents = [sorted(partition["SEN12"].percents("Dem")) for partition in chain]
    data = pandas.DataFrame(d_percents)

    ax = data.boxplot(positions=range(len(data.columns)))
    data.iloc[0].plot(style="ro", ax=ax)

    plt.savefig("base:" + str(base))

def cut_MCMC(partition):
    #print(base)
    if partition.parent is not None:
        parent_score = base ** len(partition.parent["cut_edges"])
        current_score = base ** len(partition["cut_edges"])
        ratio = base ** ( len(partition.parent["cut_edges"]) -  len(partition["cut_edges"]))
        if parent_score > current_score:
            bound = 1
        else:
            bound = ratio
            # print('bound is:', bound)
    return random.random() < bound


def cut_mcmc_go():

    # Load Data:
    graph = Graph.from_file("./PAData/PA_VTD.shp")

    election = Election("SEN12", {"Dem": "USS12D", "Rep": "USS12R"})

    starting_partition = GeographicPartition(
        graph,
        assignment="2011_PLA_1",
        updaters={
            "polsby_popper": polsby_popper,
            "cut_edges": cut_edges,
            "population": Tally("TOT_POP", alias="population"),
            "SEN12": election
        }
    )
    # df = gpd.read_file("./PAData/PA_VTD.shp")

    for base in [.05, .1, .2]:
        steps = 100
        pop_constraint = constraints.within_percent_of_ideal_population(starting_partition, 0.02)
        chain = MarkovChain(
            proposal=propose_random_flip,
            constraints=[single_flip_contiguous, pop_constraint],
            accept=cut_MCMC,
            initial_state=starting_partition,
            total_steps=steps
        )

        d_percents = [sorted(partition["SEN12"].percents("Dem")) for partition in chain]
        data = pandas.DataFrame(d_percents)

        plt.figure()
        ax = data.boxplot(positions=range(len(data.columns)))
        data.iloc[0].plot(style="ro", ax=ax)
        plt.savefig("base:" + str(base) + "steps:" + str(steps) + ".png")
        plt.close()

        # print(sorted(part["SEN12"].percents("Dem")))
        # analyze_dem_seats(chain)
