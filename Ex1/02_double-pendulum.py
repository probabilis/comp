# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:00:04 2022

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


def dSdt(S, t):
    
    l = 1
    g = 9.81
    omega = np.sqrt(g/l)
    theta_1, theta_2, p_tilde_1, p_tilde_2 = S
    
    denominator = 1 + np.sin(theta_1-theta_2)**2
    
    A_num = p_tilde_1*p_tilde_2*np.sin(theta_1 - theta_2)
    A_den = denominator
    A = A_num/A_den
    
    B_num = p_tilde_1**2 + 2*p_tilde_2**2 - 2*p_tilde_1*p_tilde_2*np.cos(theta_1 - theta_2)
    B_den = denominator**2
    B_fac = np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2)
    B = (B_num/B_den)*B_fac
    
    theta_1_der = (p_tilde_1 - p_tilde_2*np.cos(theta_1 - theta_2))/denominator
    theta_2_der = 2*p_tilde_2 - p_tilde_1*np.cos(theta_1 - theta_2)/denominator
    p_tilde_1_der = -A + B - 2*omega**2*np.sin(theta_1)
    p_tilde_2_der = A - B - omega**2*np.sin(theta_2)
    
    return np.array([theta_1_der, theta_2_der, p_tilde_1_der, p_tilde_2_der])

# ---------------------------------
t0 = 0 
t_max = 1000

epsilon = 0.01
###################################

y0_A = np.array([0, 0, 4, 2])
t0_A = t0
t_max_A = t_max
epsilon_A = epsilon

y_A, t_A = explicit_runge_kutta(dSdt, y0_A, t0_A, t_max_A, epsilon_A, a_rk4, b_rk4, c_rk4)

theta_1_A, theta_2_A, p_1_A, p_2_A = y_A

#----------------------------------

y0_B = np.array([0, 0, 0, 4])
t0_B = t0
t_max_B = t_max
epsilon_B = epsilon

y_B, t_B = explicit_runge_kutta(dSdt, y0_B, t0_B, t_max_B, epsilon_B, a_rk4, b_rk4, c_rk4)

theta_1_B, theta_2_B, p_1_B, p_2_B = y_B

#---------------------------------

def find_crossing_points(data1, data2):
    indices = []
    if abs(data1[0]) < 10**(-10):
        indices.append(0)
    for i in range(1,len(data1)-1):
        
        if data1[i] % (2 * np.pi ) > 0 and data1[i] > (2 * np.pi ) :
            print(data1[i])
            if (data1[i] % (2 * np.pi )) < 0.01 and data2[i] > 0:
                indices.append(i)

        elif (((data1[i] < 0) and (data1[i+1] > 0)) or ((data1[i] > 0) and (data1[i+1] < 0))) and data2[i] > 0:
            if abs(data1[i]) > abs(data1[i-1]):
                indices.append(i-1)
            elif abs(data1[i]) > abs(data1[i+1]):
                indices.append(i+1)
            else:
                indices.append(i)
        else:
            None

    return np.array(indices)
    
#############################################################################

relevant_points_A = find_crossing_points(theta_2_A, p_2_A)
relevant_points_B = find_crossing_points(theta_2_B, p_2_B)

def annotate_axes(ax, text, fontsize=18, color="darkgrey"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")

fig = plt.figure(figsize=(8, 10), layout="constrained")
spec = fig.add_gridspec(3, 2)

ax0 = fig.add_subplot(spec[0, 0])
annotate_axes(ax0, 'non-chaotic')

ax1 = fig.add_subplot(spec[0, 1])
annotate_axes(ax1, 'chaotic')

ax2 = fig.add_subplot(spec[1, :])
ax3 = fig.add_subplot(spec[2, :])

ax0.plot(theta_1_A[relevant_points_A], p_1_A[relevant_points_A], linestyle = 'none',
         marker = 'o', markersize = 3, label = '$\\Theta_1(t)$', color = 'midnightblue')

ax1.plot(theta_1_B[relevant_points_B], p_1_B[relevant_points_B], linestyle = 'none',
         marker = 'o', markersize = 3, label = '$\\Theta_1(t)$', color = 'salmon')

ax2.plot(t_A, theta_1_A, color = 'midnightblue', label = '$\\Theta_1(t)$')
ax2.plot(t_A, p_1_A, color = 'mediumseagreen', label = '$\\tilde{p}_1(t_n)$')
ax2.plot(t_A, theta_2_A, color = 'or ange', label = '$\\Theta_2(t)$')
ax2.plot(t_A, p_2_A, color = 'skyblue', label = '$\\tilde{p}_2(t_n)$')

ax3.plot(t_B, theta_1_B, color = 'midnightblue', label = '$\\Theta_1(t)$')
ax3.plot(t_B, p_1_B, color = 'mediumseagreen', label = '$\\tilde{p}_1(t_n)$')
ax3.plot(t_B, theta_2_B, color = 'orange', label = '$\\Theta_2(t)$')
ax3.plot(t_B, p_2_B, color = 'skyblue', label = '$\\tilde{p}_2(t_n)$')



ax0.set_xlabel('$\\Theta_1(t_n)$') ; ax0.set_ylabel('$\\tilde{p}_1(t_n)$')
ax1.set_xlabel('$\\Theta_1(t_n)$') ; ax1.set_ylabel('$\\tilde{p}_1(t_n)$')

ax2.set_xlabel('$t_n$') ; ax2.set_ylabel('$y_A(t_n)$')
ax2.set_xlabel('$t_n$') ; ax2.set_ylabel('$y_A(t_n)$')

ax1.set_xlim(-1.5,1.5)

ax0.legend() ; ax1.legend(); ax2.legend() ; ax3.legend()


ax0.title.set_text('Non-Chaotic Pendulum')
ax1.title.set_text('Chaotic Pendulum')

ax2.title.set_text('Motion $y(t_n)$ for Non-Chaotic Double-Pendulum')
ax3.title.set_text('Motion $y(t_n)$ for Chaotic Double-Pendulum')


fig.suptitle('Poincare Map | $\\Theta_2(t_n)$ = 0 and $\\tilde{p}_2(t_n)$ > 0')















    
    
    
    




