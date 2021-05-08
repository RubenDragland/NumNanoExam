# -*- coding: utf-8 -*-
"""
Created on Sat May  1 21:28:11 2021

@author: Bruker
"""


import numpy as np
from numpy.random import binomial, multinomial
from numpy.linalg import norm
import scipy as sp
from scipy.sparse import diags
from scipy.stats import binom
from scipy.integrate import simps, romb
import heapq

from time import time
from tqdm import trange, tqdm
from numba import jit, njit, vectorize

import matplotlib
from matplotlib import pyplot as plt
from celluloid import Camera

import sys
import argparse



#The general functions for the simulation.
#Tasks and plots are done in separate files. 


global beta, tau, N, h, isolationRestriction, rng
h = 0.5
beta = 0.25
tau = 10
N = 1
isolationRestriction = 1
rng = np.random.default_rng()


#%%

def fODE(y, t):
    #Returning coupled ODE as vector
    S = y[0]
    I = y[1]
    R = y[2]
    
    dSdt = - beta * I*S / N
    dIdt = beta * I*S / N - I/tau
    dRdt = I/tau
    
    f = np.array([dSdt, dIdt, dRdt])
    return f


def earlydIdt(y,t):
    I = y[1]
    odes = fODE(y,t)
    dIdt = (beta - 1/tau) * I
    odes[1] = dIdt #Simplifying the derivative of I early on.
    return odes

# 4th-order Runge-Kutta imported from Notebooks. Exactly how I would do it. 
def rk4(x, t, h, f):
    # x is coordinates (as a vector)
    # h is timestep
    # f(x) is a function that returns the derivative
    # "Slopes"
    k1  = f(x,          t)
    k2  = f(x + k1*h/2, t + h/2)
    k3  = f(x + k2*h/2, t + h/2)
    k4  = f(x + k3*h,   t + h)
    # Update time and position
    x_  = x + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return x_

#Implementing same function as done in Ex2. However, try to keep this one modular. 
#Also inspired from lectures Notebook 9. 
def runNumerics(y0, t0, tMax, h, f, method):
    steps = int(tMax/h)
    
    initialY = np.array(y0) #Making sure it is a numpy array. 
    y = np.zeros((steps +2, initialY.size) )
    t = np.zeros(steps+2)
    
    y[0, :] = y0
    t[0]    = t0
    
    Time = t0
    for i in range(steps +1):
        h = min(h, tMax-Time) #According to tips given in lecture
        y[i+1, :] = method(y[i, :], Time, h, f )
        t[i+1] = t[i] + h
        Time += h
    
    return y, t

#Inspired, but simplified from Notebook 9 shown in lectures
#All functions do stricly increase/decrease
def iterativeBisection(a, b, expression, tol = 1e-3, maxiter = 1000 ):    
    
    c = (a+b) / 2 #Midpoint
    lhs = c
    rhs = expression (c)
    err = np.abs(lhs -rhs)
    
    i = 0
    while err > tol:
        i += 1
        if i > maxiter:
            print('Failed', rhs, lhs)
            return rhs, lhs
        if (rhs-lhs) < 0:
            b = c
        else:
            a = c
        c = (a+b) / 2
        lhs = c
        rhs = expression(c)
        err = np.abs(lhs-rhs)
        print(f'Iteration {i}, error is {err:.5f}') #Progress bar from lectures. 
    print (rhs, lhs)    
    return rhs, lhs

def semiAnalyticalDeterministicS(s): 
    rNumber = beta*tau
    Sinf = np.exp( - rNumber*(1- s))
    return Sinf

def semiAnalyticalDeterministicR(r):
    rNumber = beta*tau
    Rinf = 1 - np.exp(-rNumber * r)
    return Rinf




#%%

def simulate180days(initR, IperN, h, model, f, method = rk4): 
    #General 180day simulation for several tasks
    y0 = np.array([N-IperN-initR, IperN, initR])
    t0 = 0
    tMax = 180
    
    y, t = model(y0, t0, tMax, h, f, method )
    
    return y, t


#@profile
def bigSimulate180days(initE, initI, initIa, initR, h, model, f, method ):
    
    y0 = np.array([N - initE - initR, initE, initI, initIa, initR ], dtype= np.int64 ) 
    t0 = 0
    tMax = 180 
    
    y, t = model(y0, t0, tMax, h, f, method)
    
    return y, t

#%%

#Implementation SIR model

def deltaInfection(y,t):
    dt = h 
    S = y[0]
    I = y[1]
    
    p = 1 - np.exp(-dt * beta * I / N) #Calculate probability
    changeInfected = binomial(S, p) 
    #Change given by binomial distribution.    
    return changeInfected
    
    
def deltaRecovering(y,t):
    dt = h
    I = y[1]
    
    p = 1 - np.exp(-dt / tau)
    changeInfected = binomial(I, p)
    
    return changeInfected

def binomialDerivatives(y,t):
    return np.array([deltaInfection(y,t), deltaRecovering(y,t)]) #Merged to resemble deterministic structure

#Same form as rk4, using stochastic steps
def binomialStepping(y, t, h, f):
    
    changes = f(y,t)
    deltaI= changes[0]
    deltaR = changes[1]
    
    S = y[0] - deltaI
    I = y[1] + deltaI - deltaR
    R = y[2] + deltaR
    
    y_ = np.array([S, I, R])
    
    return y_


#%%



#f, or derivatives function
#@jit(nopython = True)
#@profile
    
def multinomialDerivates(y,t):
        
    S =  y[0].astype(np.int64)
    E =  y[1].astype(np.int64)
    I =  y[2].astype(np.int64)
    Ia = y[3].astype(np.int64)
    R =  y[4].astype(np.int64)
    
    #To be able to feed numbered probabilities, sum the each S, E etc. Does also hold when S, E are numbers.
    #Calculates total population at given time the function is called (day/night)    
    Spop = np.sum(S)
    Epop = np.sum(E)
    Ipop = np.sum(I)
    Iapop = np.sum(Ia)
    Rpop = np.sum(R) 
    Npop = Spop + Epop + Ipop + Iapop + Rpop #For readability.
    
    dt = h    
    beta = 0.55
    rs = isolationRestriction
    ra = 0.1
    fs = 0.6
    fa = 0.4
    tauE = 3
    tauI = 7
    
    pExposed = 1 - np.exp(-dt*beta* (rs*Ipop + ra*Iapop)/Npop )
    pInfected = fs* (1 - np.exp(-dt/tauE) )
    pInfectedA = fa * (1- np.exp(-dt/tauE ) )
    pRecovering = 1 - np.exp(-dt/tauI)
    pRecoveringA = 1 - np.exp(-dt/tauI) #Now, these are numbers regardless of parameter dimensions
    
    deltaI, deltaIa, deltaEzero = rng.multinomial(E, [pInfected, pInfectedA, 1- pInfected - pInfectedA]).T
    deltaS = rng.binomial(S, pExposed) 
    deltaR = rng.binomial(I, pRecovering) 
    deltaRa = rng.binomial(Ia, pRecoveringA)
    
    changes = np.array( [deltaS, deltaI, deltaIa, deltaR, deltaRa] )
    
    return changes

#Same as rk4
#@jit(nopython = True)
#@profile
def multinomialStepping(y, t, h, f):
    
    changes = f(y,t)
    
    deltaS =  changes[0]
    deltaI =  changes[1]
    deltaIa = changes[2]
    deltaR =  changes[3]
    deltaRa = changes[4]
    
    S = y[0] - deltaS
    E = y[1] + deltaS - deltaI - deltaIa
    I = y[2] + deltaI - deltaR
    Ia = y[3]+ deltaIa - deltaRa
    R = y[4] + deltaR + deltaRa
    
    y_ = np.array( [S, E, I, Ia, R] )
    
    return y_
    
#%%

#Next, the commuter model should be able to run the previous for each element in the matrix. 
#Should reduce timestep to 0.5 day, so that alternating populations.
#Population N-matrix. 

#S = np.array([[], []] ) osv

#Set up initial conditions like this. New run numerics that loops in a good way. 
#Begins the same way as runNumerics
#@jit(nopython = True)
#@profile
def runCommuter(y0, t0, tMax, h, f, method):
    
    steps = int(tMax/h)
     
    y = np.zeros((steps +2, len(y0), len(y0[0]), len(y0[0,0])), dtype = np.int64)  #Has not found a better way. But is regardless not more dimensions.  *y0.shape
    t = np.zeros(steps+2)

    y[0, :] = y0
    t[0]    = t0
    
    nTowns = len(N)
    
    time = t0
    for x in range(steps +1):
        h = min(h, tMax-time) 
        #Still using the same approach as given in lectures. Now take day/night into account. 
        
        #Have decided to have h=0.5
        if x%2 : #Equals 1 is true, this should be day because i equals night is first in the matrix. 
            #Find N, S, etc by summing over columns
            for j in range(nTowns):
                #Daytime populations                
                y[x+1, :,:,j] = method(y[x, :, :, j], time, h, f )
                            
        else:
            #Find number of N, S etc by summing over rows.
            #Filter out the correct arrays for nightime, and do one step. 
            for i in range(nTowns):
                #Nighttime populations
                y[x+1, :, i, :] = method(y[x, :, i, :], time, h, f)
                
        t[x+1] = t[x] + h
        time += h  
                
    return y, t

#@profile            
def profiling():
    global N          
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
    #N = np.ones((100, 100))*100
    initE = np.zeros(N.shape)
    initE[1, 1] = 25 #The home-stayers in town 2 start of exposed
    initI = np.zeros(N.shape)
    initIa = np.zeros(N.shape)
    initR = np.zeros(N.shape)
    
    Y = [] #holds data
    for x in trange(10):
        y, t = bigSimulate180days(initE, initI, initIa, initR, h, jitjitjitINIT, jitjitjitDerivatives, jitjitjitStepping) #25 Exposed, no vaccinated
        Y.append(y)   
    return         
                
               
                
#profiling()                
    



#%%


#Tested a numba-compiled version. It runs, but for some reason, it does not manage to get any development.

#@profile
@jit(nopython = True) 
def jitjitjitDerivatives(y,t):
        
    S =  y[0]#.astype(np.int64)
    E =  y[1]#.astype(np.int64)
    I =  y[2]#.astype(np.int64)
    Ia = y[3]#.astype(np.int64)
    R =  y[4]#.astype(np.int64) 
    
    #print(S)
    
    
    #To be able to feed numbered probabilities, sum the each S, E etc. Does also hold when S, E are numbers.
    #Calculates total population at given time the function is called (day/night)    
    Spop = np.sum(S)
    Epop = np.sum(E)
    Ipop = np.sum(I)
    Iapop = np.sum(Ia)
    Rpop = np.sum(R) 
    Npop = Spop + Epop + Ipop + Iapop + Rpop #For readability.
    
    #print(Epop)
    
    dt = h    
    beta = 0.55
    rs = 1#isolationRestriction
    ra = 0.1
    fs = 0.6
    fa = 0.4
    tauE = 3
    tauI = 7
    
    pExposed = 1 - np.exp(-dt*beta* (rs*Ipop + ra*Iapop)/Npop )
    pInfected = fs* (1 - np.exp(-dt/tauE) )
    pInfectedA = fa * (1- np.exp(-dt/tauE ) )
    pRecovering = 1 - np.exp(-dt/tauI)
    pRecoveringA = 1 - np.exp(-dt/tauI) #Now, these are numbers regardless of parameter dimensions
    
    #print(pInfected)
    
    l = len(I)    
    deltaI, deltaIa, deltaEzero = np.zeros(l), np.zeros(l), np.zeros(l)
    deltaS = np.zeros(l)
    deltaR = np.zeros(l)
    deltaRa = np.zeros(l)
    for i in range(l): #, 
        deltaI[i], deltaIa[i], deltaEzero[i] = np.random.multinomial(E[i], [pInfected, pInfectedA, 1- pInfected - pInfectedA] ) #Believe this is the problem.
        deltaS[i] = np.random.binomial(S[i], pExposed) #binomial(S, pExposed) Remember to check if this works.
        deltaR[i] = np.random.binomial(I[i], pRecovering) #binomial(I, pRecovering ) 
        deltaRa[i] = np.random.binomial(Ia[i], pRecoveringA) #binomial(Ia, pRecoveringA) Update np.random to recommended library function
        
    changes = np.zeros((5, l))
    changes[0], changes[1], changes[2], changes[3], changes[4] = deltaS, deltaI, deltaIa, deltaR, deltaRa
    
    return changes

@njit
def jitjitjitStepping(y, t, h, f):
    
    changes = jitjitjitDerivatives(y,t)
    
    deltaS =  changes[0]
    deltaI =  changes[1]
    deltaIa = changes[2]
    deltaR =  changes[3]
    deltaRa = changes[4]
    
    S = y[0] - deltaS
    E = y[1] + deltaS - deltaI - deltaIa
    I = y[2] + deltaI - deltaR
    Ia = y[3]+ deltaIa - deltaRa
    R = y[4] + deltaR + deltaRa
    
    y_ = np.zeros((5, len(S)))
    y_[0], y_[1], y_[2], y_[3], y_[4] = S, E, I, Ia, R 
    
    return y_


def jitjitjitINIT(y0, t0, tMax, h, f, method):
    
    steps = int(tMax/h)
    
    y = np.zeros((steps +2, len(y0), len(y0[0]), len(y0[0,0])), dtype = np.int64)  #Has not found a better way. But is regardless not more dimensions.  *y0.shape
    #Save memory instead, nope no good idea. Or good idea, hard to implement.
    #yOld = y0
    #yNew = y0#np.zeros(y0.shape, dtype = np.int64)
    #InfectedTime = np.zeros((steps, *y0[0][0].shape), dtype = np.int64)
    #InfectedTime[0] = np.sum(yOld[2,:, :], axis = 1) + np.sum(yOld[3,:, :], axis = 1)  #Sums over rows and merges I and Ia.
    
    t = np.zeros(steps+2)
    t[0]    = t0
    
    y[0, :] = y0
    nTowns = len(N)    
    time = t0
    
    y, t = jitjitjitRUN(steps, h, tMax, nTowns, y, t, time)
    
    return y, t

@njit    
def jitjitjitRUN(steps, h, tMax, nTowns, data, t, time): 
    
    y= data
    
    for x in range(steps +1):
        h = min(h, tMax-time) #Still using the same approach as given in lectures. Now take day/night into account. 
        
        #Something wrong with kernel. Perhaps memory not working as it should. Try redefine here?
        
        #Has to have h=0.5
        if x%2 : #Equals 1 is true, this should be day because i equals night is first in the matrix. 
            #Find N, S, etc by summing over columns
            for j in range(nTowns):
                #Daytime populations                
                y[x+1, :,:,j] = jitjitjitStepping(y[x, :, :, j], time, h, jitjitjitDerivatives )
                #yNew[:,:,j] = jitjitjitStepping(yOld[:, :, j], time, h, jitjitjitDerivatives)
                #yOld[:,:, j] = yNew[:,:,j]
                
                            
        else:
            #Find number of N, S etc by summing over rows.
            #Filter out the correct arrays for nightime, and do one step. 
            #Perhaps able to not loop by using axis?
            for i in range(nTowns):
                #Nighttime populations
                y[x+1, :, i, :] = jitjitjitStepping(y[x, :, i, :], time, h, jitjitjitDerivatives)
                #yNew[:,i,:] = jitjitjitStepping(yOld[:, i, :], time, h, jitjitjitDerivatives) 
                #yOld[:,i,:] = yNew[:,i,:]
        
        #Infected[x+1] = np.sum(yNew[2,:, :], axis = 1) + np.sum(yNew[3,:, :], axis = 1)          
        t[x+1] = t[x] + h
        time += h
    return y, t
    
''' 
profiling()
tid1 = time()
profiling()
tid2 = time()
print(tid2-tid1)
'''  
 
 #Tested yesterdays jitted version. Does not seem to work... Uncertain. 
 
 
 #%%
@jit(nopython = True) 
def multitest():
    a = np.random.multinomial(10, [0.2,0.3, 0.5] ) 
    return a

#print(multitest() ) 
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
#ExamImplementation.N = N #Important as hell.
initE = np.zeros(N.shape)
initE[0, 0] = 25 #The home-stayers in town 2 start of exposed
initI = np.zeros(N.shape)
#initI[0,0] = 100000
initIa = np.zeros(N.shape)
initR = np.zeros(N.shape)

Y = [] #holds data
for x in trange(10):
    #y, t = bigSimulate180days(initE, initI, initIa, initR, h, runCommuter, multinomialDerivates, multinomialStepping) #25 Exposed, no vaccinated
    y, t = bigSimulate180days(initE, initI, initIa, initR, h, jitjitjitINIT, jitjitjitDerivatives, jitjitjitStepping)
    Y.append(y)

Y = np.array(Y) '''

#print(Y)


def calcPop(number, S):
    pop = [np.sum(x) for x in S[:, number, :]]
    return pop

def SIIaRcommuterPLOTa(t, data, nTowns, size1, size2): #Run 10 times again.
    
    fig, axes = plt.subplots(nTowns, 1, sharex = True, sharey = False, constrained_layout = True, figsize = (size1, size2))
    
    plt.suptitle(f'Commuter model epidemic', fontsize = 20)
    
    for x, prop in enumerate(data):        
        S, E, I, Ia, R = prop[:, 0], prop[:, 1], prop[:, 2], prop[:, 3], prop[:, 4]       
        
        
        for number, y in enumerate(axes):
            #Believe it is necessary to transpose
            Spop = calcPop(number, S)           
            Epop = calcPop(number, E) 
            Ipop = calcPop(number, I)
            Iapop = calcPop(number, Ia)
            Rpop = calcPop(number, R)
            
            y.plot(t, Spop, alpha = 0.5) #label = f'S{number}')
            y.plot(t, Epop, alpha = 0.5) #label = f'E{number}')
            y.plot(t, Ipop, alpha = 0.5) #label = f'I{number}')
            y.plot(t, Iapop, alpha = 0.5) #label = f'Ia{number}')
            y.plot(t, Rpop, alpha = 0.5) #label = f'R{number}')
            y.title.set_text(f'Town{number}')
            y.set_ylabel('Population', fontsize = 14)       
    y.set_xlabel('t [days]', fontsize = 14)
    
    plt.plot()
   


#SIIaRcommuterPLOTa(t, Y, len(N), 10, 20)
    
    
    
    


    
    
    
    
    

    
    
    
       
    




