# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 17:09:13 2022

@author: Sebastian
"""
import numpy as np
import matplotlib.pyplot as plt


a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])

a_eul = np.array([[0]])

b_eul = np.array([1])

c_eul = np.array([0])


def explicit_runge_kutta(F, y0, t0, t_max, epsilon, a, b, c):
    
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


def dfdt(f, t):
    l = 1
    g = 9.81
    omega = np.sqrt(g/l)
    theta, p = f
    
    return np.array([p, -omega**2*np.sin(theta)])

m = 1
l = 1
g = 9.81
omega = np.sqrt(g/l)
y0 = np.array([np.pi/3, 0])
t0 = 0
t_max = 10
epsilon = 0.05

y_rk4, t_rk4 = explicit_runge_kutta(dfdt, y0, t0, t_max, epsilon, a_rk4, b_rk4, c_rk4)



y_eul, t_eul = explicit_runge_kutta(dfdt, y0, t0, t_max, epsilon, a_eul, b_eul, c_eul)

plt.plot(t_eul, y_eul[0,:])
plt.plot(t_rk4, y_rk4[0,:])

E_t_rk4 = 0.5*m*l**2*y_rk4[1,:]**2 + m*g*l*(1 - np.cos(y_rk4[0,:]))

E_t_eul = 0.5*m*l**2*y_eul[1,:]**2 + m*g*l*(1 - np.cos(y_eul[0,:]))

#plt.plot(t_rk4, E_t_rk4, color = 'skyblue')
#plt.plot(t_eul, E_t_eul, color = 'crimson')

    