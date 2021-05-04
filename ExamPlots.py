# -*- coding: utf-8 -*-
"""
Created on Sat May  1 22:07:25 2021

@author: Bruker
"""


#Here the plot functions are to be defined and then imported to the data analysis file. 
import numpy as np
import scipy as sp

import matplotlib
from matplotlib import pyplot as plt


def deterministicPLOTa(t, data, labels, dotted, dotLabels):
    
    fig, ax = plt.subplots(constrained_layout = True, figsize = (12, 6))
    
    dots = np.linspace(0, t[-1], 50)
    for x, prop in enumerate(data.T):
        ax.plot(t, prop, linewidth = 3, alpha = 0.7, label = f'{labels[x]}')
    for x, prop in enumerate(dotted):
        ax.plot(dots, [prop]*len(dots), '--', linewidth = 3, alpha = 0.7, label = f'{dotLabels[x]}')
    ax.legend(fontsize = 14)
    ax.set_ylim(0,1)
    ax.set_ylabel('Population fraction', fontsize = 14)
    ax.set_xlabel('t [days]', fontsize = 14)
    ax.set_title(f'Deterministic epidemic development', fontsize = 20)
    plt.show()
    
def partDeterministic(ax, t, data, labels):
    data = data*100000
    for x, prop in enumerate(data.T):
        ax.plot(t, prop, '-.', linewidth = 3, label = f'{labels[x]}')
    
    
    
def deterministicInfectedPLOTb(t, data1, data2, labels):
    
    fig, ax = plt.subplots(constrained_layout = True, figsize = (12, 6))
    
    ax.plot(t, data1, linewidth = 3, alpha = 0.7, label = f'{labels[0]}' )
    ax.plot(t, data2, '--', linewidth = 3, alpha = 0.7, label = f'{labels[1]}' )
    ax.legend(fontsize = 14)
    ax.set_yscale('log')
    ax.set_ylabel('Population fraction', fontsize = 14)
    ax.set_xlabel('t [days]', fontsize = 14)
    ax.set_title(f'Deterministic Infected simplification', fontsize = 20)
    plt.show()
    
def deterministicExpPLOTd(t, data, labels):
    
    fig, ax = plt.subplots(constrained_layout = True, figsize = (12, 6))
    dots = np.linspace(0, t[-1], 50)
    for x, prop in enumerate(data):
        ax.plot(t, prop, linewidth = 3, alpha = 0.7, label = f'Vaccinated: {labels[x]}')
    ax.plot(dots, [prop[0]]*len(dots), '--', linewidth = 3, label = 'R-number = 1' )
    ax.legend(fontsize = 14)
    ax.set_yscale('log')
    ax.set_ylabel('Population fraction', fontsize = 14)
    ax.set_xlabel('t [days]', fontsize = 14)
    ax.set_title(f'Vaccination effect on infections', fontsize = 20)
    plt.show()
    
def stochasticPLOTa(tD, tS, dataDeterministic, dataStochastic, dLabels):
    
    fig, ax = plt.subplots(constrained_layout = True, figsize = (12, 6))
    
    partDeterministic(ax, tD, dataDeterministic, dLabels)
    
    for x, prop in enumerate(dataStochastic):
        S, I, R = prop.T[0], prop.T[1], prop.T[2]
        ax.plot(tS, S, alpha = 0.5)
        ax.plot(tS, I, alpha = 0.5)
        ax.plot(tS, R, alpha = 0.5)
    
    ax.set_ylim(0,1e5)
    ax.legend()
    ax.set_ylabel('Population', fontsize = 14)
    ax.set_xlabel('t [days]', fontsize = 14)
    ax.set_title(f'Stochastic epidemic development', fontsize = 20)
    plt.show()
    
    
def stochasticInfectedPLOTb(tD, tS, dataDeterministic, dataStochastic, dLabels):
    
    fig, ax = plt.subplots(constrained_layout = True, figsize = (12, 6))
    
    ax.plot(tD, 100000*dataDeterministic.T, '-.', lw = 3, label = dLabels)
    
    for x, prop in enumerate(dataStochastic):
        S, I, R = prop.T[0], prop.T[1], prop.T[2]
        ax.plot(tS, I, alpha = 0.5)
        
    ax.legend(fontsize = 14)
    ax.set_yscale('log')
    ax.set_ylabel('Population', fontsize = 14)
    ax.set_xlabel('t [days]', fontsize = 14)
    ax.set_title(f'Infection simplification', fontsize = 20)
    plt.show()
    
def probabilityPLOTc(N, prob, std ):
    
    fig, ax = plt.subplots(constrained_layout = True, figsize = (6, 6))
    
    for point, p, s in zip(N, prob, std):
        ax.errorbar(point, p, yerr = s, lw = 3, label = f'Imported: {point}, p: {p} std: {round(s, 3)}')   
    
    ax.set_ylim(0,0.5)
    ax.legend()
    ax.set_ylabel('Probability', fontsize = 14)
    ax.set_xlabel('Initially infected', fontsize = 14)
    ax.set_title(f'Probability no epidemic', fontsize = 20)
    plt.show()
    
    
def SIIaRstochasticPLOTa(tD, tS, dataDeterministic, dataStochastic, dLabels):
    
    fig, ax = plt.subplots(constrained_layout = True, figsize = (12, 6))
    
    partDeterministic(ax, tD, dataDeterministic, dLabels)
    
    for x, prop in enumerate(dataStochastic):
        S, E, I, Ia, R = prop.T[0], prop.T[1], prop.T[2], prop.T[3], prop.T[4]
        ax.plot(tS, S, alpha = 0.5)
        ax.plot(tS, E, alpha = 0.5)
        ax.plot(tS, I, alpha = 0.5)
        ax.plot(tS, Ia, alpha = 0.5)
        ax.plot(tS, R, alpha = 0.5)
    
    ax.set_ylim(0,1e5)
    ax.legend()
    ax.set_ylabel('Population', fontsize = 14)
    ax.set_xlabel('t [days]', fontsize = 14)
    ax.set_title(f'Stochastic epidemic development', fontsize = 20)
    plt.plot()
    
def stochasticExpPlOT3b(t, data, labels):
    
    fig, ax = plt.subplots(constrained_layout = True, figsize = (12, 6))
    dots = np.linspace(0, t[-1], 50)
    for x, prop in enumerate(data):
        ax.plot(t, prop, linewidth = 3, alpha = 0.7, label = f'rs: {labels[x]}')
    ax.plot(dots, [prop[0]]*len(dots), '--', linewidth = 3, label = 'R-number = 1' )
    ax.legend(fontsize = 14)
    ax.set_yscale('log')
    ax.set_ylabel('Population', fontsize = 14)
    ax.set_xlabel('t [days]', fontsize = 14)
    ax.set_title(f'Self-isolation effect', fontsize = 20)
    plt.show()
    
    
    
    
    
    
    
    
    