# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:17:50 2022

@author: Blackhole
"""

import numpy as np

def ring(N, t, E, E_off, delta, alpha, beta):
    if E_off != 0:
        off = E_off
    else:
        off = 0
            
    I = np.identity(N)
    I_off = np.identity(N) * off
    
    Delta = np.zeros((N,N))
    i = complex(0,1)
    Delta[alpha-1, alpha-1] = delta
    Delta[beta-1, beta-1] = delta
    
    
    H_R = np.zeros((N,N))
    
    for a in range(N):
        for b in range(N):
            
            if (a == (b + 1)) or (a == (b - 1)) or (a == 0 and b == N-1) or (a == N-1 and b == 0):
                creator_destroyer = np.zeros((N,N))
                creator_destroyer[a, b] = t
                H_R = H_R + creator_destroyer
    
    matrix = H_R - E * I + I_off + i*Delta
    
    return matrix


