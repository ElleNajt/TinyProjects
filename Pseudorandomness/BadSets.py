# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:55:01 2020

@author: lnajt
"""


import math
from sympy import nextprime
import secrets
import itertools
from matplotlib.pyplot import plot
import numpy as np
'''
#Note that it's not hard to construct P that produce a very dense bad set, you
just iteratively choose the value P(j) (j = 1, ... n) so that 

(j  - P(i)/(i - j))(i - j) = P(j).

It just appears to be the case that for random P, these sets tend to be sparse.
I guess this is related to what's happening inside the choice of hash function.


'''

n = 8
r = nextprime(n**6)

min_distances = []

for trial in range(100):
    
    P = { x : secrets.randbelow(r) for x in range(n)}
    
    product_set = set(itertools.product(range(n), range(n)))
    diagonal = set ( zip( range(n), range(n)))
    for x in diagonal:
        product_set.discard(x)
        
    index = list(product_set)
    
    bad_set = [ pow( i - j, r - 2, r) * ( P[i] - P[j]) % r for i,j in index]

    
    distances = set()
    for x,y in itertools.product(bad_set, bad_set):
        d = (x - y) % r
        distances.add(d)
        
    distances.remove(0)
    
    min_distance = min(list(distances))
    min_distances.append(min_distance)
print(min(min_distances))
print(np.mean(min_distances))
histogram = [ bad_set.count(i) for i in range(r)]

plot(list(range(r)) , histogram)