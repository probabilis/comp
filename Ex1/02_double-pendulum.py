
# 2 - Double Pendulum

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner


import numpy as np
import matplotlib.pyplot as plt


#############################################
#############################################
#############################################

# General Explicit Runge Kutta Method and RK4-parameters of the Butcher tableau


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


#############################################
#############################################
#############################################

# b) Solving the Double-Pendulum numerically with the RK4-Method


def Double_Pendulum(D, t, l = 1, g = 9.8067):
    """
    Returns the system of ODE's of a double pendulum with:
        * m = 1
        * l = 1
        * g = 9.8067
    The output is a numpy array
    """

    omega = np.sqrt(g/l)
    theta1, theta2, p1, p2 = D
    
    denominator = (1 + np.sin(theta1 - theta2)**2)
    theta1_der = (p1 - p2*np.cos(theta1 - theta2)) / denominator
    theta2_der = (2*p2 - p1*np.cos(theta1 - theta2)) / denominator
    
    A = (p1*p2*np.sin(theta1 - theta2)) / denominator
    B = ((p1**2 + 2*p2**2 - 2*p1*p2*np.cos(theta1 - theta2))*np.sin(theta1 - theta2)*np.cos(theta1 - theta2)) / (denominator**2) 
    
    p1_der = B - A - 2*omega**2*np.sin(theta1)
    p2_der = A - B - omega**2*np.sin(theta2)
    
    return np.array([theta1_der, theta2_der, p1_der, p2_der])


# 4 different initial conditions
y0_list = [np.array([1, 0, 0, 3]), np.array([0, 0, 4, 2]),
           np.array([0, 0, 0, 4]), np.array([0, 0, 0, 5])]

# define plotting layout
fig, axs = plt.subplots(2, len(y0_list), figsize = (20,10))


def Plotting_Double_Pendulum(y0_list):
    t0 = 0
    tmax = 60
    epsilon = 0.01
    
    for i, y0 in enumerate(y0_list):
        y, t = explicit_runge_kutta(Double_Pendulum, y0, t0, tmax, epsilon, a_rk4, b_rk4, c_rk4)
        theta1, theta2, p1, p2 = y
        
        axs_theta = axs[0, i]
        axs_p = axs[1, i]
        
        axs_theta.plot(theta1, theta2, color = 'blue')
        axs_p.plot(p1, p2, color = 'green')
    return


Plotting_Double_Pendulum(y0_list)

for ax_top, ax_bot, i in zip(axs[0], axs[-1], y0_list):
    ax_top.set_title('$\\theta_1(0) = %1.f$, $\\theta_2(0) = %1.f$, $p_1(0) = %1.f $, $p_2(0) = %1.f$' %(i[0], i[1], i[2], i[3]))
    ax_top.set_xlabel('$\\theta_1(t)$'.format(i))
    ax_bot.set_xlabel('$p_1(t)$'.format(i))
axs[0,0].set_ylabel('$\\theta_2(t)$')
axs[1,0].set_ylabel('$p_2(t)$')

fig.suptitle('Double Pendulum - Numerical Solution using RK4 with various Initial Conditions', fontsize = 24)
fig.tight_layout()
#fig.savefig('double_pendulum.png')
plt.show()


#############################################
#############################################
#############################################

# c) Poicare Maps with chaotic and non-chaotic initial conditions


def Poincare_Points(data1, data2, epsilon):
    """Function that returns all indices where data1 = 0 and data2 > 0"""
    
    indices = []
    for i in range(len(data1) - 1):
        if (((abs(data1[i]) % (2*np.pi)) < epsilon) or ((2*np.pi - (abs(data1[i]) % (2*np.pi))) < epsilon)) and data2[i] > 0:
            indices.append(i)
            
    return np.array(indices)


y0_list = [np.array([1, 0, 0, 3]), np.array([0, 0, 4, 2]),
           np.array([0, 0, 0, 4]), np.array([0, 0, 0, 5])]

#define plotting layout
fig, axs = plt.subplots(1, len(y0_list), figsize = (20,5))


def Plotting_Poincare(y0_list):
    t0 = 0
    tmax = 360
    epsilon = 0.01
    
    for i, y0 in enumerate(y0_list):
        y, t = explicit_runge_kutta(Double_Pendulum, y0, t0, tmax, epsilon, a_rk4, b_rk4, c_rk4)
        theta1, theta2, p1, p2 = y
        relevant_points = Poincare_Points(theta2, p2, epsilon)
        
        axs_poincare = axs[i]
        
        axs_poincare.plot(theta1[relevant_points], p1[relevant_points], 
                          color = 'crimson', linestyle = 'none', marker = 'o',
                          markersize = 1.5)
    return
    
    
Plotting_Poincare(y0_list)

for ax, i in zip(axs, y0_list):
    ax.set_title('$\\theta_1(0) = %1.f$, $\\theta_2(0) = %1.f$, $p_1(0) = %1.f $, $p_2(0) = %1.f$' %(i[0], i[1], i[2], i[3]))
    ax.set_xlabel('$\\theta_1(t)$'.format(i))
axs[0].set_ylabel('$p_1(t)$')


fig.suptitle('Poincare Maps - Numerical Solution using RK4 with various Initial Conditions', fontsize = 24)
fig.tight_layout()
fig.savefig('poincare_double_pendulum.png')
plt.show()


#############################################
#############################################
#############################################

# d) Verifying the conservation of energy


# With this command an offset between plot and title is set (there was an overlap)
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 


def Hamiltonian(theta1, theta2, p1_tilde, p2_tilde, m = 1, l = 1, g = 9.8067):
    """Returns the total energy of a double pendulum"""
    
    p1 = p1_tilde*m*l**2
    p2 = p2_tilde*m*l**2
    
    T = (1/(2*m*l**2))*(p1**2 + 2*p2**2 - 2*p1*p2*np.cos(theta1 - theta2))/(1 + np.sin(theta1 - theta2)**2)
    U = m*g*l*(4 - 2*np.cos(theta1) - np.cos(theta2))
    H = T + U
    
    return H


y0_list = [np.array([1, 0, 0, 3]), np.array([0, 0, 4, 2]),
           np.array([0, 0, 0, 4]), np.array([0, 0, 0, 5])]

#define plotting layout
fig, axs = plt.subplots(1, len(y0_list), figsize = (20,5))


def Plotting_Energy(y0_list):
    t0 = 0
    tmax = 60
    epsilon = 0.01
    
    for i, y0 in enumerate(y0_list):
        y, t = explicit_runge_kutta(Double_Pendulum, y0, t0, tmax, epsilon, a_rk4, b_rk4, c_rk4)
        theta1, theta2, p1_tilde, p2_tilde = y
        
        Energy = Hamiltonian(theta1, theta2, p1_tilde, p2_tilde)
        
        axs_energy = axs[i]
        
        axs_energy.plot(t, Energy, color = 'cyan')
    return
    
    
Plotting_Energy(y0_list)

for ax, i in zip(axs, y0_list):
    ax.set_title('$\\theta_1(0) = %1.f$, $\\theta_2(0) = %1.f$, $p_1(0) = %1.f $, $p_2(0) = %1.f$' %(i[0], i[1], i[2], i[3]))
    ax.set_xlabel('$t / s$'.format(i))
axs[0].set_ylabel('$E(t)$')


fig.suptitle('Double Pendulum - Verifying the Conservation of Energy', fontsize = 24)
fig.tight_layout()
#fig.savefig('poincare_double_pendulum.png')
plt.show()





















