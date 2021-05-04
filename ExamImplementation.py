# -*- coding: utf-8 -*-
"""
Created on Sat May  1 21:28:11 2021

@author: Bruker
"""


import numpy as np
from numpy.random import binomial
from numpy.linalg import norm
import scipy as sp
from scipy.sparse import diags
from scipy.stats import binom
from scipy.integrate import simps, romb
import heapq

from time import time
from tqdm import trange, tqdm
from numba import jit

import matplotlib
from matplotlib import pyplot as plt
from celluloid import Camera

import sys
import logging
import argparse


#The general functions for the simulation.
#Tasks and plots are done in separate files. 

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#logging.info('Testing')

global beta, tau, N, h, isolationRestriction
h = 0.1
beta = 2
tau = 1
N = 1
isolationRestriction = 1

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
#Also inspired from lectures. 
def runNumerics(y0, t0, tMax, h, f, method):
    steps = int(tMax/h)
    
    initialY = np.array(y0) #Making sure it is a numpy array. 
    y = np.zeros((steps +2, initialY.size)) 
    t = np.zeros(steps+2)
    
    y[0, :] = y0
    t[0]    = t0
    
    time = t0
    for i in range(steps +1):
        h = min(h, tMax-time) #According to tips given in lecture
        y[i+1, :] = method(y[i, :], time, h, f )
        t[i+1] = t[i] + h
        time += h
    
    return y, t

#Inspired, but simplified from Notebook 9 shown in lectures
#All functions do stricly increase. Begin at 0 and exp captures the linear. 
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

#Remember holes in terms of initI and vaccinacted
def bigSimulate180days(initE, initR, h, model, f, method ):
    
    y0 = np.array([N - initE - initR, initE, 0, 0, initR ])
    t0 = 0
    tMax = 180
    
    y, t = model(y0, t0, tMax, h, f, method)
    
    return y, t

#%%

#Implementation SIR model

def deltaInfection(y,t):
    dt = h #Uncertain about this one 
    S = y[0]
    I = y[1]
    
    p = 1 - np.exp(-dt * beta * I / N) #Calculate probability
    changeInfected = binomial(S, p) #float(binom.rvs(S, p) ) #Change given by binomial distribution.
    
    return changeInfected
    
    
def deltaRecovering(y,t):
    dt = h
    I = y[1]
    
    p = 1 - np.exp(-dt / tau)
    changeInfected = binomial(I, p) #float( binom.rvs(I, p) )
    
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
def multinomialDerivates(y,t):
        
    S =  y[0]
    E =  y[1]
    I =  y[2]
    Ia = y[3]
    R =  y[4]
    
    dt = h    
    beta = 0.55
    rs = isolationRestriction
    ra = 0.1
    fs = 0.6
    fa = 0.4
    tauE = 3
    tauI = 7
    
    pExposed = 1 - np.exp(-dt*beta* (rs*I + ra*Ia)/N )
    pInfected = fs* (1 - np.exp(-dt/tauE) )
    pInfectedA = fa * (1- np.exp(-dt/tauE ) )
    pRecovering = 1 - np.exp(-dt/tauI)
    pRecoveringA = 1 - np.exp(-dt/tauI)
    
    rng = np.random.default_rng()
    deltaI, deltaIa, deltaEzero = rng.multinomial(E, [pInfected, pInfectedA, 1- pInfected - pInfectedA])    
    deltaS = binomial(S, pExposed)
    deltaR = binomial(I, pRecovering ) 
    deltaRa = binomial(Ia, pRecoveringA)
    
    changes = np.array( [deltaS, deltaI, deltaIa, deltaR, deltaRa] )
    
    return changes

#Same as rk4
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
    


    
    
    
    


    
    
    
    
    

    
    
    
       
    




