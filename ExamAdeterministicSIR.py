# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:32:10 2021

@author: Bruker
"""


import numpy as np
import scipy as sp

from time import time
from tqdm import trange, tqdm
from numba import jit

import matplotlib
from matplotlib import pyplot as plt

from ExamImplementation import fODE, earlydIdt, rk4, runNumerics, simulate180days
from ExamImplementation import  iterativeBisection, semiAnalyticalDeterministicS, semiAnalyticalDeterministicR

from ExamPlots import deterministicPLOTa, deterministicInfectedPLOTb, deterministicExpPLOTd
import ExamImplementation

global beta, tau, N, h #Issue with global variables for functions defined elsewhere. 
ExamImplementation.h = 0.01
ExamImplementation.beta = 0.25
ExamImplementation.tau = 10 
ExamImplementation.N = 1
infectedRatioInit = 1e-4

h = ExamImplementation.h


#%%
#Deterministic epidemic

#y, t = simulate180days(0, infectedRatioInit, h, runNumerics, fODE, rk4 )
#lims = np.array( [iterativeBisection(0, 1, semiAnalyticalDeterministicR) [0], iterativeBisection(0,1, semiAnalyticalDeterministicS) [0] ])
labels = ['Susceptible', 'Infected', 'Recovered']
dotLabel = ['lim R', 'lim S']

#deterministicPLOTa(t, y, labels, lims, dotLabel)

#np.save(f'dSIRtaskParametersh{h}.npy', y)
#np.save(f'dSIRtaskT{h}.npy', t)


#%%

#Simplified

#I = y.T[1]

#redo, t = simulate180days(0, infectedRatioInit, h, runNumerics, earlydIdt, rk4)
#Iearly = redo.T[1]
l = ['Infected', 'I simplified']

#deterministicInfectedPLOTb(t, I, Iearly, l)

#np.save(f'IntensityB{h}.npy', I)
#np.save(f'IntensitySimplifiedB{h}.npy', Iearly )

#CHECK THIS SHIIIT should not diverge. 

#%%

#Flattening the curve

#See graphically that beta = 0.25 is too large. Do a bisection scheme to determine the optimal. 
#Go bit lower, evaluate and try again

#ExamImplementation.beta = 0.25
    
def peakAfterRestrictions(beta):
    ExamImplementation.beta = beta
    y, t = simulate180days(0, infectedRatioInit, h, runNumerics, fODE, rk4)
    I = y.T[1]
    peak = np.max(I)
    return peak


#To find max allowed value, inspired from lectures. Adjusted from the prior bisection function  since the desired value is now known. 
def bisectionMethod(desiredPeak, a, b, expression, tol = 1e-3, maxiter = 100):
    peakA = peakAfterRestrictions(a)
    peakB = peakAfterRestrictions(b)
    
    c = (a+b)/2
    peakC = peakAfterRestrictions(c)
    err = np.abs(desiredPeak - peakC )
    
    i = 0
    while err > tol or peakC > 0.2:
        i += 1
        if i > maxiter:
            print('Failed', peakC, c)
            return peakC, c
        if (desiredPeak-peakC)*(desiredPeak- peakA) < 0:
            b = c
            peakB = peakC.copy()
        else:
            a = c
            peakA = peakC.copy()
        
        c = (a+b)/2
        peakC = peakAfterRestrictions(c)
        err = np.abs(desiredPeak - peakC )
        
        print(f'Iteration {i}, error is {err:.5f}') #Progress bar from lectures. 
    
    return peakC, c
        

#maxPeak, maxBeta = bisectionMethod(0.2, 0, 0.25, peakAfterRestrictions)

#Can easily be calculated analytically from the derivative in the expression. EDIT: NOT. 
#print(maxPeak, maxBeta)
#Look up similar methods to the bisection method. Secant seems nice, not Newton has to take the derivative. 


#%%

#Vaccinated population

#Initially vaccinated
initV = np.array( [0, 1e-6, 1e-3, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95] )

def calculateInfectedGivenR(initV):
    Infected = []
    for x, R in enumerate(initV):
        
        
        y0 = np.array([ExamImplementation.N-infectedRatioInit-R, infectedRatioInit, R])
        t0 = 0
        tMax = 25
        
        y,t = runNumerics(y0, t0, tMax, h, fODE, rk4)
        Infected.append(y.T[1])
    
    Infected = np.array(Infected)
    return Infected, t
        
Infections, t = calculateInfectedGivenR(initV)
vaccineLabels = [ '0', '1e-6', '1e-3', '0.1', '0.2', '0.4', '0.6', '0.8', '0.95']

deterministicExpPLOTd(t, Infections, vaccineLabels)

#As seen from the plot, approx. 60% of the population should be vaccinated in order to prevent an outbreak  with 1e-4 imported infected. 


#%%

#Toggle rs (Selfisolating) to make reduce the infections
    



    
    
    





