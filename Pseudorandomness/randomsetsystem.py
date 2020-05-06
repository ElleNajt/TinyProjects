

import numpy as np
import itertools
from matplotlib import pyplot as plt



def weight(collection, weight_function):
    sum = 0
    for x in collection:
        sum += weight_function[x]

    return sum


def min_unique(family, weight_function):
    weights = np.array([ weight(collection, weight_function) for collection in family])
    min = np.min(weights)
    locations = np.where(weights == min)[0]
    #print(weights)
    #print(min)
    #print(locations)
    #print(len(locations))
    return len(locations), min


def plot(family, weight_function, tag):
    weights = np.array([ weight(collection, weight_function) for collection in family])
    print("mean", np.mean(weights))
    f = plt.figure()
    plt.hist(weights, bins = 1000)
    f.savefig(tag)
    plt.close()


for k in [1,2,3,4,5]:
    universe_size = 100
    universe = list(range(universe_size))
    family_size = 10000
    cleaned = True

    tag =  "cleaned" + str(cleaned) + "weights_unisize" + str(universe_size) + "K" + str(k) + "FAMSIZE" + str(family_size)

    weight_function = {}
    for x in universe:
        weight_function[x] = x + 1


    success = 0
    trials = 1



    for i in range(trials):
        random_family = []

        for i in range(family_size):
            collection = tuple(np.random.choice(universe, k, replace = False))
            random_family.append(collection)

        #clean of duplicates

        if cleaned == True:
            random_family_cleaned = list(set(random_family))
            random_family = []
            for collection in random_family_cleaned:
                random_family.append( np.array(collection))

        #print(random_family)
        unique, min = min_unique(random_family, weight_function)
        if unique == 1:
            success += 1
    print(success/trials)

    plot(random_family, weight_function, tag)
