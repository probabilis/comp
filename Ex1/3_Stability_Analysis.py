
# 3 - Stability Analysis

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

# System of ODE's


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
#fig.savefig('poincare_double_pendulum.png')
plt.show()






















