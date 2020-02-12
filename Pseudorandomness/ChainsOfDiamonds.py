# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:52:30 2020

@author: lnajt
"""



import itertools
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import secrets
import sympy
from sympy import nextprime


n = 20
d = 0

block_1 = list(range(n, n + d+ 1))
block_2 = list(range(n + d + 1, n + 2*d + 1 + 1))
block_3 = list(range( n + 2*d + 2, n + 3*d + 3))
block_4 = list(range(n + 3*d + 3, n + 4*d + 4))
p = nextprime(max(block_4))
block_1_updated = block_1
block_2_updated = block_2
block_3_updated = block_3
block_4_updated = block_4

conflict = 0
for t in range(10000):
    a = secrets.randbelow(p)
    while a == 0:
        a = secrets.randbelow(p)
    
    block_1_updated = [(a * x) % p for x in block_1]
    block_2_updated = [(a * x) % p for x in block_2]    
    block_3_updated = [(a * x) % p for x in block_3]    
    block_4_updated = [(a * x) % p for x in block_4]
    
    left_hand = min(block_1_updated) + min(block_2_updated)
    right_hand = min(block_3_updated) + min(block_4_updated)
    #print(block_1_updated, block_2_updated, block_3_updated, block_4_updated)
    if (left_hand ==  right_hand):
        conflict += 1
        print(a)
print(conflict)