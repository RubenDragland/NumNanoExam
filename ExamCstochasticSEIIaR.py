# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:55:03 2021

@author: Bruker
"""

import numpy as np
from numpy.random import binomial
import scipy as sp

from time import time
from tqdm import trange, tqdm
from numba import jit

import matplotlib
from matplotlib import pyplot as plt

from ExamImplementation import  runNumerics, simulate180days, deltaInfection, deltaRecovering, binomialDerivatives, binomialStepping
from ExamImplementation import fODE, earlydIdt, rk4
from ExamImplementation import  iterativeBisection, semiAnalyticalDeterministicS, semiAnalyticalDeterministicR
from ExamImplementation import multinomialDerivates, multinomialStepping, bigSimulate180days

from ExamPlots import deterministicPLOTa, deterministicInfectedPLOTb, deterministicExpPLOTd, stochasticPLOTa, partDeterministic, stochasticInfectedPLOTb
from ExamPlots import probabilityPLOTc, SIIaRstochasticPLOTa, stochasticExpPlOT3b
import ExamImplementation

global beta, tau, N, h #Issue with global variables for functions defined elsewhere. 
ExamImplementation.h = 1 #Initally believe the stepping should be between 1 and 3 days, due to incubation time.
ExamImplementation.beta = 0.25
ExamImplementation.tau = 10 
ExamImplementation.N = 100000
infectedRatioInit = 10

h = ExamImplementation.h

#%%

#SIIaR 10 simulation, all in the same plot
'''
Y = []
for x in trange(10):
    y, t = bigSimulate180days(25, 0, h, runNumerics, multinomialDerivates, multinomialStepping) #25 Exposed, no vaccinated
    Y.append(y)

Y = np.array(Y)
np.save(f'2Ca10simulations180daysh{h}.npy', Y)
np.save(f'2Ca10timeh{h}.npy', t)

dataDeterministic= np.load(f'dSIRtaskParametersh0.01.npy')
tD = np.load ('dSIRtaskT0.01.npy')

dLabels = ['deterministic S', 'deterministic I', 'deterministic R']

SIIaRstochasticPLOTa(tD, t, dataDeterministic, Y, dLabels)
'''
#Forskjøvet, forventet pga inkubasjon.


#%%

#Retrieving a global variable linked to rs, Isolating infected with symptoms. 
ExamImplementation.isolationRestriction = 1

isolations = np.linspace(0, 1, 11)
isolations = np.linspace(0.30, 0.40, 11)

#Same approach as 1d. 
def calculateInfectedGivenIsolation(isolations):
    infected = []
    for x, rs in enumerate(isolations):       
        ExamImplementation.isolationRestriction = rs
        
        y0 = np.array([ExamImplementation.N - 25, 25, 0, 0, 0 ])
        t0 = 0
        tMax = 180
        
        y,t = runNumerics(y0, t0, tMax, h, multinomialDerivates, multinomialStepping)
        infected.append(y.T[1])
    
    infected = np.array(infected)
    return infected, t

infections, t = calculateInfectedGivenIsolation(isolations)
isolationLabels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
isolationLabels = ['0.3', '0.31', '0.32', '0.33', '0.34', '0.35', '0.36', '0.37', '0.38', '0.39', '0.40']

#np.save('isolation3B180daysZOOMED.npy', infections)
#np.save('isolationtime180daysZOOMED.npy', t)


stochasticExpPlOT3b(t, infections, isolationLabels)


#rs 0.4 is below R1 almost 25 days.
#Tested 180 days, changes every time. a bit of luck or unluck, 0.3 seems fine. 
#About half of the 30s is always below R1. Suggestion, do confidence-interval for 0.35, see how many times below R1 in 100 runs. 
