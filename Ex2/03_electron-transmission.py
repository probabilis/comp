# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:17:50 2022

@author: Blackhole
"""

import numpy as np
import matplotlib.pyplot as plt
from ring_function import ring
from gauss_seidel import gauss_seidel
from numpy import asarray


##programm for solving the electron transmission through molecular transport system 

alpha = [1,1] 
beta = [3,4]

N = 6

t = -2.6
delta = 0.5

E_1 = -6
E_2 = 6

E_ran = np.arange(E_1, E_2 + 1, 0.1)
E_off = 20

limit = 1000
tol = 1e-8

def system_calc(N,t,E_ran,delta,alpha,beta, limit, tol):
    
    
    E_out = np.zeros((len(E_ran),len(alpha)), dtype = object)


    #########
    #calculation of coefficents matrices E for different E_i and a/b
   
    for j, E in enumerate(E_ran):
        k = -1
        for a, b in zip(alpha,beta):
            k += 1
            E_out[j][k] = ring(N, t, E, E_off, delta, a, b)
            #print(E_out[j][k])
            

    #########
    #calculation of green functions
    I = np.identity(N)

    G_Rab = np.zeros((len(E_ran),len(alpha), N, N), dtype = object)
    #print(E_out.shape)

    for E in range(len(E_ran)):
        for i in range(len(alpha)):
            for n in range(len(I)):

                g_i = gauss_seidel(E_out[E][i], I[n], limit, tol)

                G_Rab[E][i][n] = g_i


    #print(G_Rab.shape)

    return G_Rab, E_out


GR_matrix, E_out = system_calc(N,t,E_ran,delta,alpha,beta, limit, tol)
#print(GR_matrix))

#check?
#K = np.dot(E_out[0][1],(GR_matrix[0][1]).T)
#print(K)

k = -1
for i in GR_matrix[:]:
    k += 1
    plt.plot(E_ran[k], abs(i[0][0][2])**2, 'o', color = 'salmon', label = '$\\alpha = 1, \\beta = 3$')
    #plt.plot(E_ran[k], abs(i[1][0][3])**2, 'x', color = 'aqua',label =  '$\\alpha = 1, \\beta = 4$')


plt.title('Transmission probability $T_{\\alpha\\beta}(E)$')
plt.ylabel('$T_{\\alpha\\beta}(E)$')
plt.xlabel('E')
plt.xlim(-10,10)




E_out = np.zeros((len(E_ran),len(alpha)), dtype = object)


E_off = 0

j = -1
for E in E_ran:
    j = j + 1
    k = -1
    for a, b in zip(alpha,beta):
        k = k + 1
        E_out[j][k] = ring(N, t, E, E_off, delta, a, b)
        #print(E_out[j][k])


fig, ax = plt.subplots(1,1)

for i in range(len(E_ran)):
    g_i = np.linalg.inv(E_out[i][0])
    ax.plot(E_ran[i], abs(g_i[0][2])**2, '.')
    #ax.plot(E_ran[i], abs(g_i[0][3])**2, '.')

ax.set_xlim(-10,10)

#ax.set_ylim(0,0.1)

fig.show()

#ev_i = np.linalg.eig()
