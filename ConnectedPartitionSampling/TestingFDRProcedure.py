import EnumeratingConnectedPartitions
import numpy as np
import scipy.stats
import networkx as nx
import countforests
'''
To explore whether making assumptions such as:

1) There is a true low rank flat that glues together means.
--> It makes sense to be agnostic about means being the same for samples_as_partitions
that are not in the same block though.
-->
2) The true neighborhood of relevant partitions is larger.


Inferentially, I think the question is :
--> Which assumptions make sense for discovery recommendations.
------> Can this ever be tested?

------

Something that can be tested: learning the true connected partition.

'''



def estimate_posterior_low_rank(graph, rank = 10, sample_size = 1000, prior, X_samples, Y_samples, partition, block_indicator):
    #Estimates P( block_indicator, assignment | X, Y)

    graph = countforests.initialize_weights(graph)

    partitions = []
    for i in range(sample_size):
        partition = countforests.produce_sample(graph, rank, 1000)
        partitions.append(partition)



    return 0

def likelihood_approximation(prior, X_samples, Y_samples,
                            partition, block_indicator):
    #Uses the Laplace approximation from Vo's thesis to approximate
    #log f( X,Y | partition, block_indicator )


    return 0

def likelihood_marginalizing_covariances(X_samples, Y_samples, mu_X, mu_Y, block_indicator, partition):
    #This calculates f(X,Y | mu_X, mu_Y, Delta, Psi)



def posterior_approximation(prior, X_samples, Y_samples):
    #Produce samples, calculate likelihoods, etc.

    return 0

def produce_sample(partition, block_indicator,
                M_X, M_Y
                , graph, delta_0, mu_0, tau_squared
                , sigma_squared, df, A, B):
    #Notation following Vo's Thesis, Newton et al.
    #Partition stored as set of edges internal to a given blocks
    #block_indicator is stored as dictionary on the components

    components = list(nx.connected_components( nx.edge_subgraph(graph, partition)))
    #use this as the ordering on components


    if len(block_indicator.keys()) != len(components):
        print("block indicator not correct length")
        return False

    #These are the means assigned to each cluster.

    delta = {}
    phi = {}
    nu = {}
    mu_X = []
    mu_Y = []
    for block in components:
        phi[block] = np.random.normal(mu_0, tau_squared)
        #This is following the notation in Vo
        if block_indicator(block) == 0:
            delta[block] = 0
        else:
            delta[block] = np.random.normal(delta_0, sigma_squared)
        nu[block] = delta[block] + phi[block]
        mu_X.append(phi[block])
        mu_Y.append(nu[block])

    U = scipy.stats.invwishart(df, A)
    W = scipy.stats.invwishart(df, B)

    X_sample = []
    for i in range(M_X):
        X_m = np.random.normal(mu_X, U)
        X_sample.append(X_m)
    Y_sample = []
    for i in range(M_Y):
        Y_m = np.random.normal(mu_Y, W)
        Y_sample.append(Y_m)

    return X_sample, Y_sample
