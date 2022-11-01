# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:58:24 2022

@author: Sebastian
"""


import numpy as np

A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0., 3., -1., 8.]])

b = np.array([6.0, 25.0, -11.0, 15.0])

x = np.zeros_like(b)

limit = 100
tol = 1e-8


def gauss_seidel(A, x, b, limit, tol):
    
    u = 0
    n = len(A)
    
    for k in range(n):
        if (np.sum(abs(A[k, :k])) + np.sum(abs(A[k, k+1:]))) < abs(A[k,k]):
            u += 1
    if u == n: 
    
        x_sol = []
        for lim in range(limit):
            x_sol.append(x.copy())
            x_new = np.zeros_like(x)
        
            for i in range(n):
                sum_1 = np.sum(A[i, :i]*x_new[:i])
                sum_2 = np.sum(A[i, i + 1 :]*x[i + 1 :])
                x_new[i] = (b[i] - sum_1 - sum_2)/A[i,i]
            if np.allclose(x, x_new, rtol= tol):
                break
            if lim == (limit-1):
                print('Procedure does not converge! Set your limit higher.')
            x = x_new
        return np.array(x_sol)
    
    else:
        text = 'The convergence criteria for matrix A is not met!'
        return text


print(gauss_seidel(A, x, b, limit, tol))