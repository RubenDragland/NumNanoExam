# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:17:39 2021

@author: Bruker
"""

import numpy as np
from numpy.random import binomial
import scipy as sp
from pandas import read_csv


from time import time
from tqdm import trange, tqdm
from numba import jit, njit

import matplotlib
from matplotlib import pyplot as plt

from ExamImplementation import  runNumerics, simulate180days, deltaInfection, deltaRecovering, binomialDerivatives, binomialStepping
from ExamImplementation import fODE, earlydIdt, rk4
from ExamImplementation import  iterativeBisection, semiAnalyticalDeterministicS, semiAnalyticalDeterministicR
from ExamImplementation import multinomialDerivates, multinomialStepping, bigSimulate180days, runCommuter

from ExamPlots import deterministicPLOTa, deterministicInfectedPLOTb, deterministicExpPLOTd, stochasticPLOTa, partDeterministic, stochasticInfectedPLOTb
from ExamPlots import probabilityPLOTc, SIIaRstochasticPLOTa, stochasticExpPlOT3b, SIIaRcommuterPLOTa, calcPop, SIIaRnorwayPLOTa, nationWideOutbreakPLOTe
import ExamImplementation

global beta, tau, N, h #Issue with global variables for functions defined elsewhere. 
ExamImplementation.h = 0.5 #Now do half a day timestep. 
ExamImplementation.beta = 0.25
ExamImplementation.tau = 10 
ExamImplementation.N = 100000
infectedRatioInit = 10

h = ExamImplementation.h
ExamImplementation.isolationRestriction = 1

#%%

#Assume that 25 in (0,0) in town 1 (Not commuters) are initally exposed. Will create a small delay. 

'''
N = np.array( [[9000, 1000], [200, 99800] ])
ExamImplementation.N = N #Important as hell.
initE = np.zeros(N.shape)
initE[0, 0] = 25 #The home-stayers in town 1 start of exposed
initR = np.zeros(N.shape)

Y = [] #holds data
for x in trange(10):
    y, t = bigSimulate180days(initE, initR, h, runCommuter, multinomialDerivates, multinomialStepping) #25 Exposed, no vaccinated
    Y.append(y)

Y = np.array(Y)
#np.save(f'2Da10simulations180daysh{h}.npy', Y)
#np.save(f'2Da10timeh{h}.npy', t)


#np.save(f'2Da10TEST180daysh{h}.npy', Y)
#np.save(f'2Da10TESTtimeh{h}.npy', t)

SIIaRcommuterPLOTa(t, Y, 2)

'''
#Notes for b
#Tests done to ensure
#No outbreak in big city if no commuting. See fig. 
#No obreak in big city if commuting inbetween, but all in every commute. 
#When almost all the people commute into the big city, the epidemic is instanteneous. 


#%%

#Bigger commuting system

'''
N = np.array( [[198600, 100, 100, 100, 100, 1000, 0, 0, 0, 0],
               [500, 9500, 0, 0, 0, 0, 0, 0, 0, 0],
               [500, 0, 9500, 0, 0, 0, 0, 0, 0, 0],
               [500, 0, 0, 9500, 0, 0, 0, 0, 0, 0],
               [500, 0, 0, 0, 9500, 0, 0, 0, 0, 0],
               [1000, 0, 0, 0, 0, 498200, 200, 200, 200, 200],
               [0, 0, 0, 0, 0, 1000, 19000, 0, 0, 0],
               [0, 0, 0, 0, 0, 1000, 0, 19000, 0, 0],
               [0, 0, 0, 0, 0, 1000, 0, 0, 19000, 0],
               [0, 0, 0, 0, 0, 1000, 0, 0, 0, 19000],               
               ]) #DO NOT DELETE
ExamImplementation.N = N #Important as hell.
initE = np.zeros(N.shape)
initE[1, 1] = 25 #The home-stayers in town 2 start of exposed
initR = np.zeros(N.shape)

Y = [] #holds data
for x in trange(10):
    y, t = bigSimulate180days(initE, initR, h, runCommuter, multinomialDerivates, multinomialStepping) #25 Exposed, no vaccinated
    Y.append(y)

Y = np.array(Y)
#np.save(f'2Ea10simulations180daysh{h}.npy', Y)
#np.save(f'2Ea10timeh{h}.npy', t)


#Y = np.load(f'2Ea10simulations180daysh{h}.npy')
#t = np.load(f'2Ea10timeh{h}.npy')
SIIaRcommuterPLOTa(t, Y, len(N))
'''

#%%

#Norway Supercomputer

def readFHI(filename):
    pd = read_csv(filename)
    arr = np.array(pd) #pd.to_numpy()
    
    return arr, pd


#N, pd = readFHI('population_structure_Norway.csv')

#np.save('NorwayInit.npy', N)

#N = np.load('NorwayInit.npy')
'''
ExamImplementation.N = N #Important as hell.
initE = np.zeros(N.shape)
initE[0, 0] = 50 #The home-stayers in town 2 start of exposed
initR = np.zeros(N.shape)

Y = [] #holds data
for x in trange(10):
    y, t = bigSimulate180days(initE, initR, h, runCommuter, multinomialDerivates, multinomialStepping) 
    Y.append(y)

Y = np.array(Y)
np.save(f'2EbNORWAY10simulations180daysh{h}.npy', Y)
np.save(f'2EbNORWAY10timeh{h}.npy', t)
'''

#Y = np.load(f'2EbNORWAY10simulations180daysh{h}.npy')
#t = np.load(f'2EbNORWAY10timeh{h}.npy')
#SIIaRcommuterPLOTa(t, Y, 10)

#%%



@jit(nopython = True)
def countInfected(t, Y):
    valueSimulation = np.zeros((len(Y), len(t)))
    nTowns = len(Y[0][0][0])
    
    for simNb, prop in enumerate(Y):
        
        S, E, I, Ia, R = prop[:, 0], prop[:, 1], prop[:, 2], prop[:, 3], prop[:, 4] 
        
        #Probabibly a quicker way... Wanted to jit something...
            
        for townNumber in range(nTowns):
            Ipop = np.array([np.sum(x) for x in I[:, townNumber, :]])
            Iapop = np.array([np.sum(x) for x in Ia[:, townNumber, :]])   
            Itot = Ipop + Iapop
            for timestep, infectedTime in enumerate(Itot):
                if infectedTime > 10:
                    valueSimulation[simNb, timestep ] += 1
    return t, valueSimulation


#t, valueSimulation = countInfected(t, Y)




#np.save('2EbActualPlot.npy', valueSimulation)
#np.save('2EbActualTime.npy', t)


#counts = np.load('2EbActualPlot.npy')
#t = np.load('2EbActualTime.npy')


#nationWideOutbreakPLOTe(t, counts)

#%%

#Modified reduced travel


def reduceTravel(N):
    
    for rowN, pop in enumerate(N):
        
        conservation1 = np.sum(N[rowN])        
        homeOffice = round(np.sum(N[rowN]*0.9) - N[rowN, rowN]*0.9 )
        diagCity = N[rowN, rowN] + homeOffice
        N[rowN] = N[rowN]//10 #Decided to use % to ensure the number of people are conserved. Note, no measure taken if fewer than 10 in city. 
        N[rowN, rowN] = diagCity
        conservation2 = np.sum(N[rowN])
        if (conservation1 != conservation2):
            N[rowN, rowN] += conservation1-conservation2           
        
    return N
    
test = np.array([[100, 111], 
                 [500, 1000]])

N = np.array( [[198600, 100, 100, 100, 100, 1000, 0, 0, 0, 0],
               [500, 9500, 0, 0, 0, 0, 0, 0, 0, 0],
               [500, 0, 9500, 0, 0, 0, 0, 0, 0, 0],
               [500, 0, 0, 9500, 0, 0, 0, 0, 0, 0],
               [500, 0, 0, 0, 9500, 0, 0, 0, 0, 0],
               [1000, 0, 0, 0, 0, 498200, 200, 200, 200, 200],
               [0, 0, 0, 0, 0, 1000, 19000, 0, 0, 0],
               [0, 0, 0, 0, 0, 1000, 0, 19000, 0, 0],
               [0, 0, 0, 0, 0, 1000, 0, 0, 19000, 0],
               [0, 0, 0, 0, 0, 1000, 0, 0, 0, 19000],               
               ]) #DO NOT DELETE

print(reduceTravel(N))
'''
norwayRestrictions = np.load('NorwayInit.npy')
tic = time()
norwayRestrictions = reduceTravel(norwayRestrictions)
tac = time()
print(tac-tic) '''

#np.save('NorwayRestricted.npy', norwayRestrictions)

#Now restricted population structure. 
#N = np.load('NorwayRestricted.npy')
'''
ExamImplementation.N = N #Important as hell.
initE = np.zeros(N.shape)
initE[0, 0] = 50 #The home-stayers in town 2 start of exposed
initR = np.zeros(N.shape)

Y = [] #holds data
for x in trange(10):
    y, t = bigSimulate180days(initE, initR, h, runCommuter, multinomialDerivates, multinomialStepping) 
    Y.append(y)

Y = np.array(Y)
np.save(f'2EcNORWAYRESTRICTED10simulations180daysh{h}.npy', Y)
np.save(f'2EbNORWAYRESTRICTED10timeh{h}.npy', t)
'''
'''
Y = np.load(f'2EcNORWAYRESTRICTED10simulations180daysh{h}.npy')
t = np.load(f'2EbNORWAYRESTRICTED10timeh{h}.npy')
      
t, counts = countInfected(t, Y)

  
nationWideOutbreakPLOTe(t, counts)  
'''

#Peak shifted by 25 days. Does not quite believe it... It was not much...
#Test the others as well...
        
        
        
        
                  
                
                       
        
    
