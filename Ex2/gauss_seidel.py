# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

# Gauss-Seidel module which returns the solution vector

import numpy as np


def gauss_seidel(A, b, limit, tol):
    
    x = np.zeros_like(b)
    n = len(A)
    
    x_sol = []
    for lim in range(limit):
        x_sol.append(x.copy())
        x_new = np.zeros_like(x, dtype = np.cdouble)
        
        for i in range(n):
            sum_1 = np.sum(A[i, :i]*x_new[:i])
            sum_2 = np.sum(A[i, i + 1 :]*x[i + 1 :])
            x_new[i] = (complex(b[i],0) - sum_1 - sum_2) / A[i,i]
        if np.allclose(x, x_new, rtol= tol):
            break
        if lim == (limit-1):
            print('Procedure does not converge! Set your limit higher.')
        x = x_new
    return x_sol[-1]