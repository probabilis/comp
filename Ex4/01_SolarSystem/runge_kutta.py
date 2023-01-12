# -*- coding: utf-8 -*-
#General Runge-Kutta Function Modul for CP Project for solving linear-diff.eq systems 

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

import numpy as np

def explicit_runge_kutta(F, y0, t0, t_max, epsilon, a, b, c):
    """
    input parameters:

    F(y,t) = array of 1 order d.e. functions
    y0 = initial conditions in y
    t0 = initial condition in t
    t_max = maximal time value 
    epsilon = incremental size of differentation
    a, b, c = coefficients for specific method (RK4, Euler,.. )

    output:
    y, t = solution vector  in y and time vector t

    """
    nr_y = y0.size
    
    t = np.arange(t0, t_max, epsilon)
    nr_t = t.size
    
    y = np.zeros((nr_y, nr_t))
    y[:,0] = y0
    
    d = c.size
    k = np.zeros((nr_y, nr_t, d))
    
    for n in range(nr_t - 1):
        
        for i in range(0,d):
            delta_k = 0
            for j in range(d):  
                delta_k += a[i,j]*F(k[:,n,j], t[n] + c[j]*epsilon)
                
            k[:,n,i] = y[:,n] + epsilon*delta_k
            
        delta_y = 0
        for j in range(d):
            delta_y += b[j]*F(k[:,n,j], t[n] + epsilon*c[j])
                              
        y[:,n+1] = y[:,n] + epsilon*delta_y
        
    return y, t

