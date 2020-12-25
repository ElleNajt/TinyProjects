# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:59:03 2020

@author: lnajt

TODO:
    1. The 'uniform connected partition of 3x3x3 grid-graph' oracle can also be used as yet another proposal function for MCMC on connected partitions.
    2. I was thinking that another reasonable Bayesian model would be to use Swendson-Wang samples from the random cluster model to block means. However, because blocking should be transitive and this model wouldn't produce transitive results, maybe it would make more sense to have the random cluster model sample impose zeros in the covariance matrix drawn from Wishart. The comments here ( https://stats.stackexchange.com/questions/246140/simulating-from-an-inverse-wishart-with-constraints )  suggests that such conditioned Wishart matrices can be sampled from (G-Wishart matrices).
    
    But this is sort of the opposite of what we want -- since want things in the same component to be more correlated, not indepndent.
    
    You could also work with zeros given by edges in the complment of the RCM sample, but this seems to turn quickly into making the blocks independent, which didn't seem consistent with the data.'
    
    3. It seems possible to make a reasonable guess at the underlying partition. 
For instance, assuming for simplicity that covariance matrices are the identity matrix instead of coming from inverse Wishart, for each edge {v,w} we can decide whether v and w are blocked by estimating the correlation of those two random variables (zero in the unblocked case and some fixed positive number in the blocked case). Doing this for every edge and correcting it to give a connected partition (e.g. by taking the components of the induced graph) seems like a reasonable algorithm. 
Perhaps something similar happens in the case that the covariance comes from an inverse Wishart distribution? I'd guess that the negative/positive correlations coming from the random covariance matrix cancel out, and one is left with a clear correlation only in the case that the variables are blocked.
Even if the goal is to sample from the posterior, such as for estimating local false discovery rates, my intuition would be that seeding the run at such a guess would lead to a better sample.
    
    4. Is the posterior distribution convex in some sense over the set of partitions, or at least unimodal and decaying quickly as one moves away from the generative partition?
5. I'm trying to think of a way to benchmark these different approaches. I think the most concrete thing to do would be to try to recover the original partition given some samples, as you all did in section 6 of the supplemental material. 
"""

import random 
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random
from numba import njit

def visualizing_wishart():
    

    
    for scale in [1,10]:
        df = 20*scale - 3
        A = [[30*scale,0],[0,10*scale]]
        U = scipy.stats.invwishart(df, A)
        for i in range(3):
            m1 = (0,0)
            s1 = U.rvs(1)
            k1 = multivariate_normal(mean=m1, cov=s1)
            xlim = (-3, 3)
            ylim = (-3, 3)
            xres = 100
            yres = 100
    
            x = np.linspace(xlim[0], xlim[1], xres)
            y = np.linspace(ylim[0], ylim[1], yres)
            xx, yy = np.meshgrid(x,y)
    
            # evaluate kernels at grid points
            xxyy = np.c_[xx.ravel(), yy.ravel()]
            zz = k1.pdf(xxyy) 
    
            # reshape and plot image
            img = zz.reshape((xres,yres))
            plt.imshow(img); plt.show()

# @njit            
def A_matrix(data_matrix, mu):
    
    #The whole thing should be vectorized
    A = np.zeros([len(mu), len(mu)])
    
    for x in data_matrix:
        A += np.outer(x - mu, x - mu)
        
    return A

# @njit
def integrate(data_matrix, Phi, allocation_matrix, mu_partition, tau_sq, df, num_observations, num_samples):

    """
    

    Parameters
    ----------
    Phi : TYPE
        DESCRIPTION.
    L : matrix
        the allocation matrix -- determines which block each node belongs to
        if u is a vector of blockwise values, Lu spreads it to each node
        DESCRIPTION.
    mu_0 : TYPE
        DESCRIPTION.
    nu : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    num_samples : integer
    
    Returns
    -------
    None.

    """
    total_value = 0
    
    D = np.linalg.inv(Phi +np.cov(data_matrix.T))
    x_bar = np.mean(data_matrix,0)
    for i in range(num_samples):
        mu = np.random.normal(mu_partition, tau_sq)
        # A = A_matrix(data_matrix, np.matmul(allocation_matrix, mu))
        # integrand = np.linalg.det( Phi + A)**( -.5 * (df + num_observations))
        L_mu = np.matmul(allocation_matrix, mu)
        integrand = (1 + num_observations * np.matmul(( x_bar - L_mu), np.matmul(D, (x_bar - L_mu))) )**( -.5 * (df + num_observations))
        total_value += integrand
    
    return total_value / num_samples


def generate_allocation_matrix(nodes, partition):
    
    # nodes = [1,2,3,4]
    # partition = [[1,2], [3],[4]]
    
    L = np.zeros([len(nodes), len(partition)])
    for i in range(len(nodes)):
        for j in range(len(partition)):
            if nodes[i] in partition[j]:
                L[i][j] = 1
    return L
    

def generate_data(nodes, partition):
    """
    

    Parameters
    ----------
    graph : TYPE
        DESCRIPTION.
    partition : TYPE
        as a set of frozen sets of vertices of graph.

    Returns
    -------
    None.

    """
    
    p_0 = .1
    mu_0 = 0
    tau_sq = 1
    df = 10
    num_observations = 5
    
    
    nodes = list(range(10))
    partition = [frozenset([0,1,2]), frozenset([3]),frozenset([4,5,6,7,8,9])]
    allocation_matrix = generate_allocation_matrix(nodes, partition)
    
    
    num_blocks = len(partition)
    num_nodes  = len(nodes)
    
    
    Phi = np.identity(num_nodes)
    
    ## Generate mu
    mu_partition = []
    for x in partition:
        mu_partition.append( random.normalvariate(mu_0, tau_sq))
    
    ## Generate covariance matrices
    U = scipy.stats.invwishart(df, Phi).rvs(1)
    mu_X = np.matmul(allocation_matrix, mu_partition)
    data_matrix = np.random.multivariate_normal(mu_X, U, num_observations)
    
    integrate(data_matrix, Phi, allocation_matrix, mu_0 * np.zeros(num_blocks), tau_sq, df, num_observations, 1000)
    
    print("try importance sampling?")
    
generate_data()
