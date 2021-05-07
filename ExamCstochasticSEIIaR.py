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
from numba import jit, njit

import matplotlib
from matplotlib import pyplot as plt

from ExamImplementation import  runNumerics, simulate180days, deltaInfection, deltaRecovering, binomialDerivatives, binomialStepping
from ExamImplementation import fODE, earlydIdt, rk4
from ExamImplementation import  iterativeBisection, semiAnalyticalDeterministicS, semiAnalyticalDeterministicR
from ExamImplementation import multinomialDerivates, multinomialStepping, bigSimulate180days, jitjitjitDerivatives

from ExamPlots import deterministicPLOTa, deterministicInfectedPLOTb, deterministicExpPLOTd, stochasticPLOTa, partDeterministic, stochasticInfectedPLOTb
from ExamPlots import probabilityPLOTc, SIIaRstochasticPLOTa, stochasticExpPlOT3b, SIIaRconfidencePlotb
import ExamImplementation

global beta, tau, N, h #Issue with global variables for functions defined elsewhere. 
ExamImplementation.h = 0.1 #Initally believe the stepping should be between 1 and 3 days, due to incubation time.
ExamImplementation.beta = 0.25
ExamImplementation.tau = 10 
ExamImplementation.N = 100000
infectedRatioInit = 10 #Not used

h = ExamImplementation.h

#%%

#SIIaR 10 simulation, all in the same plot
'''
Y = []
for x in trange(10):
    y, t = bigSimulate180days(25,0,0, 0, h, runNumerics, multinomialDerivates, multinomialStepping) #25 Exposed, no vaccinated
    Y.append(y)

Y = np.array(Y)
#np.save(f'2Ca10simulations180daysh{h}.npy', Y)
#np.save(f'2Ca10timeh{h}.npy', t)

dataDeterministic= np.load(f'dSIRtaskParametersh0.01.npy')
tD = np.load ('dSIRtaskT0.01.npy')

dLabels = ['deterministic S', 'deterministic I', 'deterministic R']

SIIaRstochasticPLOTa(tD, t, dataDeterministic, Y, dLabels)
'''
#Forskjøvet, forventet pga inkubasjon osv.
#Kind of ruined due to commuter model... Fix bigsimulate... Solved.


#%%

#Retrieving a global variable linked to rs, Isolating infected with symptoms. 
ExamImplementation.isolationRestriction = 1

isolations = np.linspace(0, 1, 11)
isolations = np.linspace(0.30, 0.40, 11)

#Same approach as 1d. 
def calculateInfectedGivenIsolation(isolations):
    infected = []
    for x, rs in tqdm(enumerate(isolations), total = len(isolations)):       
        ExamImplementation.isolationRestriction = rs
        
        y0 = np.array([ExamImplementation.N - 25, 25, 0, 0, 0 ])
        t0 = 0
        tMax = 180
        
        y,t = runNumerics(y0, t0, tMax, h, multinomialDerivates, multinomialStepping)
        infected.append(y.T[2] +y.T[3]) #Actually exposed. FIX THIS Not all of them are infected. (Perhaps solved now)
    
    infected = np.array(infected)
    return infected, t

#infections, t = calculateInfectedGivenIsolation(isolations)
isolationLabels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
isolationLabels = ['0.3', '0.31', '0.32', '0.33', '0.34', '0.35', '0.36', '0.37', '0.38', '0.39', '0.40']

#np.save('isolation3B180daysZOOMED.npy', infections)
#np.save('isolationtime180daysZOOMED.npy', t)


#stochasticExpPlOT3b(t, infections, isolationLabels)

#NOTE that the drop is explained by the fact that the graph shows exposed people. 


#rs 0.4 is below R1 almost 25 days.
#Tested 180 days, changes every time. a bit of luck or unluck, 0.3 seems fine. 
#About half of the 30s is always below R1. Suggestion, do confidence-interval for 0.35, see how many times below R1 in 100 runs. 

def confidenceIntervalIsolation(trials, rs):
    ExamImplementation.isolationRestriction = rs
    
    expResults = np.zeros(10)
    
    for runN, outbreak in enumerate(expResults):
        success = 0
        y0 = np.array([ExamImplementation.N-25, 25, 0, 0, 0])
        t0 = 0
        tMax = 180
        for x in trange(trials):
            y,t = runNumerics(y0, t0, tMax, h, multinomialDerivates, multinomialStepping)
            infected = y.T[2] + y.T[3]
            if np.all(infected <= 2*25): #R-number=1 if everyone infected infects one.
                success += 1
                continue 
        expResults[runN] = success
    meanValue = np.mean(expResults)
    stdValue = np.std(expResults)   
    
    return expResults, meanValue, stdValue 


results, average, stDeviation = confidenceIntervalIsolation(100, 0.33)


#print(average, stDeviation )


#np.save('confidenceResult100h01.npy', results)
#results = np.load('confidenceResult100h01.npy')
#average = 90.5
#stDeviation = 3.5

SIIaRconfidencePlotb(results, average, stDeviation)