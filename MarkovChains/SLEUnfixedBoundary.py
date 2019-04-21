
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import cProfile
from SLEExperiments import integral_disc
from Facefinder import planar_dual, draw_with_location

D = integral_disc(10)
for v in D.nodes():
    D.node[v]["pos"] = D.node[v]["coord"]
dual = planar_dual(D)
draw_with_location(dual)