# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:24:42 2021

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

from ExamPlots import deterministicPLOTa, deterministicInfectedPLOTb, deterministicExpPLOTd, stochasticPLOTa, partDeterministic, stochasticInfectedPLOTb
from ExamPlots import probabilityPLOTc
import ExamImplementation

global beta, tau, N, h #Issue with global variables for functions defined elsewhere. 
ExamImplementation.h = 0.1 #Initally believe the stepping should be between 1 and 3 days, due to incubation time.
ExamImplementation.beta = 0.25
ExamImplementation.tau = 10 
ExamImplementation.N = 100000
infectedRatioInit = 10

h = ExamImplementation.h

#%%


#SIR 10 simulation, all in the same plot

Y = []
for x in trange(10):
    y, t = simulate180days(0, infectedRatioInit, h, runNumerics, binomialDerivatives, binomialStepping )
    Y.append(y) 
#lims = np.array( [iterativeBisection(0, 1, semiAnalyticalDeterministicR) [0], iterativeBisection(0,1, semiAnalyticalDeterministicS) [0] ])
#labels = ['Susceptible', 'Infected', 'Recovered']
#dotLabel = ['lim R', 'lim S']

Y = np.array(Y)
#np.save(f'2Ba10simulations180daysh{h}.npy', Y)
#np.save(f'2Ba10timeh{h}.npy', t)

#Y = np.load('2Ba10simulations180daysh{h}.npy')
#tS = np.load('2Ba10timeh{h}.npy')
dataDeterministic= np.load(f'dSIRtaskParametersh0.01.npy')
tD = np.load ('dSIRtaskT0.01.npy')


dLabels = ['deterministic S', 'deterministic I', 'deterministic R']

stochasticPLOTa(tD, t, dataDeterministic, Y, dLabels)



#%%

#Confirming the early development

#simplified = np.load('IntensitySimplifiedB0.01.npy')

#stochasticInfectedPLOTb(tD, t, simplified, Y, 'Simplified I'  ) 

#%%

#C 

#Outbreak dissapears if I = 0 instead of growing exp.
#Probability is only the share of successes when trial goes to a very large number or inf.
#Uncertainty given by random numbers and number of trials. Look up. If 1000 trials, should be able to determine % +- 0.5%

#Plot probaility as a function of N init Infected. The value is the average value, y-error is the std. 

#Only necessary to look at the first 10-20 days

#Tested by a mistake with deterministic. Got expected results. 

def miracleProbabilityAnalysis(trials):

    initI = np.linspace(1, 10, 10)
    expectedMiracles = []
    stdMiracles = []
    for outbreak in initI:
        miracles = 0
        y0 = np.array([ExamImplementation.N-outbreak-0, outbreak, 0])
        t0 = 0
        tMax = 25
        for x in trange(trials):
            y,t = runNumerics(y0, t0, tMax, h, binomialDerivatives, binomialStepping)
            infected = y.T[1]
            if np.any(infected == 0):
                miracles += 1
                continue                
        
        expectedMiracles.append( miracles ) #Expected value binomial distribution
        stdMiracles.append(np.sqrt(miracles*(1-miracles/trials)/trials**2)) #sqrt of variance binomial distribution
    
    return np.array(expectedMiracles)/trials, np.array(stdMiracles), initI


def deviationFromTrials(trials):
    samples = np.zeros((10,10))
    for test in trange(10):
        probability, stdNOT, initI = miracleProbabilityAnalysis(trials)
        samples[test, :] = probability
    
    averages = np.mean(samples, axis = 0)        
    std = np.std(samples, axis = 0)
    
    return averages, std, initI

trials = 1000
#p, std, N = miracleProbabilityAnalysis(trials)

#p, std, N = deviationFromTrials(trials)

#np.save(f'ProbabilityP2ct{trials}.npy', p)
#np.save(f'ProbabilitySTD2c{trials}.npy', std)

#p= np.load('ProbabilityP2ct1000.npy')
#std = np.load('ProbabilitySTD2c1000.npy')
#N = np.linspace(1, 10, 10)

#probabilityPLOTc(N, p, std)    

#Uncertain about the std, but but. See answer.
