# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:27:56 2020

@author: lnajt
"""


import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './')

import PhaseTransitionExperimentCode


mu = 2.63815853
bases = [2* mu]
pops = [.1]

PhaseTransitionExperimentCode.run_experiment(bases,  pops)
