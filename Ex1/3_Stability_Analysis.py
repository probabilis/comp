# -*- coding: utf-8 -*-

# 3 - Stability Analysis

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner


import numpy as np
import matplotlib.pyplot as plt
#importing the global General-Explicit RK algorithm from the project folder
from runge_kutta import explicit_runge_kutta 
#importing the global Double-Pendulum differential equations 
from double_pendulum_de import Double_Pendulum

#RK4-parameters of the Butcher tableau


a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])

#############################################
#############################################
#############################################

# a) + b)
# Calculating two (each) trajectories in phase space with slightly deviating initial conditions
# Plotting delta(t) for three chaotic and non chaotic cases 


def Delta(theta1, theta2, p1, p2, theta1_, theta2_, p1_, p2_):
    
    a = (theta1 - theta1_)**2
    b = (theta2 - theta2_)**2
    c = (p1 - p1_)**2
    d = (p2 - p2_)**2
    
    delta = np.sqrt(a + b + c + d)
    
    return delta


# 2D-List with 6 cases for two slightly deviating initial conditions
y0_list = [[np.array([0, 0, 4, 2]), np.array([0, 0, 3.99, 2.01])],
           [np.array([0, 0, 2, 1]), np.array([0, 0, 2, 0.99])],
           [np.array([1, 0, 0, 3]), np.array([1, 0, 0, 3.01])],
           [np.array([0, 0, 0, 4]), np.array([0, 0, 0, 4.01])],
           [np.array([0, 0, 0, 6]), np.array([0, 0, 0, 6.01])],
           [np.array([0, 4, 0, 7]), np.array([0, 4, 0, 6.99])]]


#define plotting layout
fig, axs = plt.subplots(1, len(y0_list), figsize = (30,5))


def Plotting_Stability(y0_list):
    t0 = 0
    tmax = 60
    epsilon = 0.01
    
    for i, y0 in enumerate(y0_list):
        y, t = explicit_runge_kutta(Double_Pendulum, y0[0], t0, tmax, epsilon, a_rk4, b_rk4, c_rk4)
        theta1, theta2, p1, p2 = y
        
        y_, t_ = explicit_runge_kutta(Double_Pendulum, y0[1], t0, tmax, epsilon, a_rk4, b_rk4, c_rk4)
        theta1_, theta2_, p1_, p2_ = y_
        
        delta = Delta(theta1, theta2, p1, p2, theta1_, theta2_, p1_, p2_)
        
        axs_stability = axs[i]
        
        axs_stability.plot(t, delta, color = 'magenta')
    return
    
    
Plotting_Stability(y0_list)
k = 0
for ax, i in zip(axs, y0_list):
    k = k + 1
    ax.set_xlabel('$t / s$'.format(i))
    if k < 4:
        ax.title.set_text('non-chaotic')
    else:
        ax.title.set_text('chaotic')


axs[0].set_ylabel('$\delta(t)$')


fig.suptitle('Stability Analysis - 3 non-chaotic and 3 chaotic initial conditions', fontsize = 24)
fig.tight_layout()
#fig.savefig('double-pendulum_stability-analysis.png')
plt.show()






















