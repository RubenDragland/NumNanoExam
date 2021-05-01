# -*- coding: utf-8 -*-
"""
Created on Sat May  1 21:28:11 2021

@author: Bruker
"""


import numpy as np
from numpy.linalg import norm
import scipy as sp
from scipy.sparse import diags
from scipy.sparse.linalg import spilu, LinearOperator, bicgstab, gcrotmk
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


#The general functions for the simulation.
#Tasks and plots are done in separate files. 

