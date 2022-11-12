# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:58:24 2022

@author: Blackhole
"""
import numpy as np

def gauss_seidel(A, b, limit, tol, weight):
    x = np.zeros_like(b)
    u = 0
    n = len(A)

    for k in range(n):
        if (np.sum(np.abs(A[k, :k])) + np.sum(np.abs(A[k, k+1:]))) < np.abs(A[k,k]):
            u += 1
            
    u = n
    
    if u == n: 
    
        x_sol = []
        for lim in range(limit):
            x_sol.append(x.copy())
            x_new = np.zeros_like(x, dtype = np.cdouble)
        
            for i in range(n):
                sum_1 = np.sum(A[i, :i]*x_new[:i])
                sum_2 = np.sum(A[i, i + 1 :]*x[i + 1 :])
                x_new[i] = weight * (complex(b[i],0) - sum_1 - sum_2) / A[i,i]
            if np.allclose(x, x_new, rtol= tol):
                break
            if lim == (limit-1):
                print('Procedure does not converge! Set your limit higher.')
            x = x_new
        return x_sol[-1]
    
    else:
        text = 'The convergence criteria for matrix A is not met!'
        return text