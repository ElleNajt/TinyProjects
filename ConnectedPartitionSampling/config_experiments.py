# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:03:57 2020

@author: lnajt
"""


import Edge_Flip_Markov_Chain_cy
import Edge_Flip_Markov_Chain
import time
size = 15
MC_steps = 2000000
MC_temperature = .7
fugacities = [1]

start_time = time.time()
Edge_Flip_Markov_Chain_cy.test_grid_graph(size, MC_steps, MC_temperature, fugacities)
print("time_used", time.time() - start_time)
start_time = time.time()

#Edge_Flip_Markov_Chain.test_grid_graph(size, MC_steps, MC_temperature, fugacities)
#print("time_used", time.time() - start_time)