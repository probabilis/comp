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


m = 1
l = 1
g = 1

# ---------------------------------

y0_A = np.array([0, 0, 4, 2])
t0_A = 0
t_max_A = 10
epsilon_A = 0.01

y_A, t_A = explicit_runge_kutta(dSdt, y0_A, t0_A, t_max_A, epsilon_A, a_rk4, b_rk4, c_rk4)

theta_1_A, theta_2_A, p_1_A, p_2_A = y_A

#----------------------------------

y0_B = np.array([0, 0, 0, 4])
y0_B = np.array([1, 0, 0, 4])
t0_B = 0
t_max_B = 10
epsilon_B = 0.01

y_B, t_B = explicit_runge_kutta(dSdt, y0_B, t0_B, t_max_B, epsilon_B, a_rk4, b_rk4, c_rk4)

theta_1_B, theta_2_B, p_1_B, p_2_B = y_B

#---------------------------------

def find_crossing_points(data1, data2):
    indices = []
    if abs(data1[0]) < 10**(-10):
        indices.append(0)
    for i in range(1,len(data1)-1):

        if data1[i] // (np.pi) > 0:

            #print(data1[i])
            data1[i] = data1[i] - data1[i] // (np.pi)  * np.pi
            #print(data1[i])
            if data1[i] < 0.01 and  data2[i] > 0:
          
                if abs(data1[i]) > abs(data1[i-1]):
                    indices.append(i-1)
                elif abs(data1[i]) > abs(data1[i+1]):
                    indices.append(i+1)
                else:
                    indices.append(i)
         

        elif data1[i] // (- np.pi ) > 0:
            #print(data1[i])
            data1[i] = abs(data1[i]) - abs(data1[i] // np.pi) * np.pi
            #print(data1[i])
            if abs(data1[i]) < 0.01 and  data2[i] > 0:
                
                if abs(data1[i]) > abs(data1[i-1]):
                    indices.append(i-1)
                elif abs(data1[i]) > abs(data1[i+1]):
                    indices.append(i+1)
                else:
                    indices.append(i)
               
        else:
            if (((data1[i] < 0) and (data1[i+1] > 0)) or ((data1[i] > 0) and (data1[i+1] < 0))) and data2[i] > 0:
                
                if abs(data1[i]) > abs(data1[i-1]):
                    indices.append(i-1)
                elif abs(data1[i]) > abs(data1[i+1]):
                    indices.append(i+1)
                else:
                    indices.append(i)
    return np.array(indices)

relevant_points_A = find_crossing_points(theta_2_A, p_2_A)
relevant_points_B = find_crossing_points(theta_2_B, p_2_B)

def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")

fig = plt.figure(figsize=(8, 10), layout="constrained")
spec = fig.add_gridspec(3, 2)

ax1_A = fig.add_subplot(spec[0, :])
annotate_axes(ax1_A, '$\\Theta(t_n)_A$ $\\tilde{p}_A(t_n)$')

ax1_B = fig.add_subplot(spec[1, :])
annotate_axes(ax1_B, '$\\Theta(t_n)_B$ $\\tilde{p}_B(t_n)$')

ax2 = fig.add_subplot(spec[2, 0])
annotate_axes(ax2, '$\\Theta_A(t)$')

ax3 = fig.add_subplot(spec[2, 1])
annotate_axes(ax3, '$\\Theta_B(t)$')

ax1_A.plot(t_B, theta_1_A, color = 'midnightblue', label = '$\\Theta_A(t)$')
ax1_A.plot(t_B, theta_2_A, color = 'mediumseagreen', label = '$\\tilde{p}_A(t_n)$')

ax1_B.plot(t_B, theta_1_A, color = 'salmon', label = '$\\Theta_A(t)$')
ax1_B.plot(t_B, theta_2_A, color = 'gold', label = '$\\tilde{p}_A(t_n)$')

ax2.plot(theta_1_A[relevant_points_A], p_1_A[relevant_points_A], linestyle = 'none',
         marker = 'o', markersize = 3, label = '$\\Theta_A(t)$', color = 'midnightblue')

ax3.plot(theta_1_B[relevant_points_B], p_1_B[relevant_points_B], linestyle = 'none',
         marker = 'o', markersize = 3, label = '$\\Theta_B(t)$', color = 'salmon')


ax1_A.set_xlabel('$t_n$') ; ax1_A.set_ylabel('$y(t_n)$')
ax1_B.set_xlabel('$t_n$') ; ax1_B.set_ylabel('$y(t_n)$')

ax2.set_xlabel('$\\Theta(t_n)_A$') ; ax2.set_ylabel('$\\tilde{p}_A(t_n)$')
ax3.set_xlabel('$\\Theta(t_n)_B$') ; ax3.set_ylabel('$\\tilde{p}_B(t_n)$')


ax1_A.legend() ; ax1_B.legend() ; ax2.legend() ; ax3.legend()

fig.suptitle('Poincare Map | $\\Theta_B(t_n)$ = 0 and $\\tilde{p}_B(t_n)$ > 0')















    
    
    
    





