# 02 - The split operator method

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from crank_nicolson_module import crank_nicolson
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import time

# a) Free Particle with Split Operator Method

# Important note: h-bar is set to 1 in this exercise

Nx = 17000
dx = 0.1
Nt = 4000
dt = 0.1
m = 1

def split_operator(Nx, dx, Nt, dt, psi0, V, V0):
    
    """
    input parameters:
        
    * Nx   ... number of spatial grid points
    * dx   ... distance between two adjacent grid points
    * Nt   ... maximum number of time steps
    * dt   ... size of time step
    * psi0 ... wave function at t = 0 as the initial condition
    * V    ... a function of the potential
    * V0   ... height of the potential steps 
    
    output paramters:
    
    * psi  ... wave function psi(x,t) as 2D-array, rows = time, columns = space 
    """
    
    i = complex(0,1)
    x_values = np.arange(-Nx*dx/3, 2*Nx*dx/3, dx)
    
    # Dimensions of the psi-output, Nt = number of steps -> Nt+1 points in time 
    x_dim = Nx
    t_dim = Nt + 1
    # Length of the space interval
    L = dx*(Nx-1)
    
    # Initialising the psi-array, cdouble to handle complex wave functions
    psi = np.zeros((t_dim, x_dim), dtype = np.cdouble)
    
    # 1) Initial and boundary conditions
    psi[0] = psi0(x_values)
    psi[0][0] = 0
    psi[0][Nx-1] = 0
    
    # Loop over all times
    for n in range(0, Nt):
        
        term_1 = np.exp(-(i/2)*V(x_values, V0)*dt)
        
        y1 = np.exp(-(i/2)*V(x_values, V0)*dt)*psi[n]
        # Fourier transformation
        y1f = fft(y1)
        
        # Positive and negative frequencies for reflections
        q1 = np.linspace(0, 2*np.pi/L*(Nx/2-1), int(Nx/2)) 
        # Negative frequencies need to be slightly shifted           
        q2 = np.linspace(-2*np.pi/L*Nx/2, -2*np.pi/L, int(Nx/2))   
        q = np.concatenate((q1,q2))
        
        y2 = np.exp(-i*q**2/(2*m)*dt)*y1f
        # Inverse Fourier transformation
        y2f_inv = ifft(y2)
        
        psi[n+1] = term_1*y2f_inv
        
    return psi


def gaussian_wave_packet(x, sigma = 20, x_0 = 0, q = 2.2, m = 1):
    
    """
    Gaussian wave packet at t = 0 with:
    * sigma = 20
    * x_0   = 0
    * q     = 2.20
    * m     = 1
    """
    
    i = complex(0,1)
    norm = 1/(np.sqrt(sigma*np.sqrt(np.pi)))
    exp_1 = np.exp(-(x - x_0)**2/(2*sigma**2))
    exp_2 = np.exp(i*q*x)
    
    return norm*exp_1*exp_2


# Potential V(x) = 0 --> free particle
def V_free(x, V0):
    V = 0
    return V


V0_free = 0


#############################################
#############################################
#############################################

# b) Comparison of Split Operator Method with Crank-Nicolson


# For Derivation see 02_b_Verification
def analytical_solution(x, t, sigma = 20, x_0 = 0, q = 2.2, m = 1):
    
    v = q/m
    delta = t/(m*sigma**2)
    norm = 1/(np.sqrt(np.pi)*sigma*np.sqrt(1 + delta**2))
    exponential = np.exp(-(x - v*t)**2/(sigma**2*(1 + delta**2)))
    
    return norm*exponential


# (x,t) for plotting
x_values = np.arange(-Nx*dx/3, 2*Nx*dx/3, dx)
time_values = np.arange(0, Nt*dt + dt, dt)


# Solving with Split Operator Method
psi_free_split = split_operator(Nx, dx, Nt, dt, gaussian_wave_packet, V_free, V0_free)
probability_density_free_split = np.absolute(psi_free_split)**2

# Solving with Crank-Nicolson
psi_free_ck = crank_nicolson(Nx, dx, Nt, dt, gaussian_wave_packet, 
                                              V_free, V0_free, m)
probability_density_free_ck = np.absolute(psi_free_ck)**2

# Times where the solutions are compared to the analytical solution
time_list = [0, 100, 200, 400]

# Define plotting layout
fig, axs = plt.subplots(1, len(time_list), figsize = (24,5), dpi = 300)

# Plotting Function
def plot_comparison(time):
    
    for i, t in enumerate(time):
        
        axs_comp = axs[i]
        axs_comp.plot(x_values, probability_density_free_split[int(t/dt)], 
                      color = 'red', linewidth = 0.8, label = 'Split Operator')
        axs_comp.plot(x_values, probability_density_free_ck[int(t/dt)], color =
                      'darkgreen', linewidth = 0.8, linestyle = 'dashdot', label = 'Crank-Nicolson')
        axs_comp.plot(x_values, analytical_solution(x_values, t), linestyle = 
                      '--', color = 'royalblue', linewidth = 0.8, label = 'Analytical')
        axs_comp.legend(loc = 'upper left')
    return

plot_comparison(time_list)

for ax, t in zip(axs, time_list):
    ax.set_title('$t = %1.f$' %t, fontsize = 16)
    ax.set_xlabel('$x$', fontsize = 14)
axs[0].set_ylabel('$|\psi(x,t)|^2$')


fig.suptitle('Probability Density - Split Operator, Crank-Nicolson and Analytical', fontsize = 24)
fig.tight_layout()
plt.show()


#############################################
#############################################
#############################################

# c) Solution of Schrödinger equation for V1 and V2

Nx = 8000
dx = 0.1
Nt = 1500
dt = 0.1
m = 1
a = 100
b = 200
d = 10
V0 = [1.5, 2.0, 2.5]

def psi0_scatter(x, sigma = 20, x_0 = 0, q = 2, m = 1):
    
    """
    Gaussian wave packet at t = 0 with:
    * sigma = 20
    * x_0   = 0
    * q     = 2
    * m     = 1
    """
    
    i = complex(0,1)
    norm = 1/(np.sqrt(sigma*np.sqrt(np.pi)))
    exp_1 = np.exp(-(x - x_0)**2/(2*sigma**2))
    exp_2 = np.exp(i*q*x)
    
    return norm*exp_1*exp_2

# heaviside function see https://de.wikipedia.org/wiki/Heaviside-Funktion
def heaviside(x, x0):
    return np.where(x < x0, 0.0, 1.0)


# Definition of the two potentials V_1 and V_2
def V_1(x, V0):
    V1 = V0*(heaviside(x, a) - heaviside(x, a+d))
    return V1


def V_2(x, V0):
    V2 = V_1(x, V0) + V0*(heaviside(x, b) - heaviside(x, b+d))
    return V2

# (x,t) for plotting
x_values = np.arange(-Nx*dx/3, 2*Nx*dx/3, dx)
time_values = np.arange(0, Nt*dt + dt, dt)

# Animation function that generates gif's for different V0
def animation_wave_scattering(V0):
    
    for v0 in V0:
        psi_V1 = split_operator(Nx, dx, Nt, dt, psi0_scatter, V_1, v0)
        psi_V2 = split_operator(Nx, dx, Nt, dt, psi0_scatter, V_2, v0)
        probability_density_V1 = np.absolute(psi_V1)**2
        probability_density_V2 = np.absolute(psi_V2)**2
        
        fig1 = plt.figure(figsize = (10,7), dpi = 300) 
        axis1 = plt.axes(xlim =(x_values[0], x_values[-1]), ylim =(-0.01, 0.06))
        plt.title('Propagating Probability Density with $V_1(x)$ and $V_0$ = ' 
                  + str(v0), fontsize = 16)
        plt.xlabel('x', fontsize = 14)
        plt.ylabel('$|\psi(x,t)|^2$', fontsize = 14)
        plt.vlines(a, 0, 0.04, color = 'darkgreen', linewidth = 1)
        plt.vlines(a + d, 0, 0.04, color = 'darkgreen', linewidth = 1)
        plt.hlines(0.04, a, a + d, color = 'darkgreen', linewidth = 1)
        plt.grid(alpha = 0.5)
        line, = axis1.plot([], [], marker = '.', lw = 0) 
        
        def init(): 
            line.set_data([], [])
            return line,
           
        def animate1(i):
            x = x_values
            y = probability_density_V1[i]
            line.set_data(x, y)
              
            return line,
        
        anim1 = FuncAnimation(fig1, animate1, init_func = init,
                             frames = len(time_values), interval = 10, blit = True)
        
        anim1.save('Wave_V1_split' + str(v0) + '.gif', dpi=100, 
                  writer=PillowWriter(fps=30))
        
        fig2 = plt.figure(figsize = (10,7), dpi = 300) 
        axis2 = plt.axes(xlim =(x_values[0], x_values[-1]), ylim =(-0.01, 0.06))
        plt.title('Propagating Probability Density with $V_2(x)$ and $V_0$ = ' 
                  + str(v0), fontsize = 16)
        plt.xlabel('x', fontsize = 14)
        plt.ylabel('$|\psi(x,t)|^2$', fontsize = 14)
        plt.vlines(a, 0, 0.04, color = 'darkgreen', linewidth = 1)
        plt.vlines(a + d, 0, 0.04, color = 'darkgreen', linewidth = 1)
        plt.hlines(0.04, a, a + d, color = 'darkgreen', linewidth = 1)
        plt.vlines(b, 0, 0.04, color = 'darkgreen', linewidth = 1)
        plt.vlines(b + d, 0, 0.04, color = 'darkgreen', linewidth = 1)
        plt.hlines(0.04, b, b + d, color = 'darkgreen', linewidth = 1)
        plt.grid(alpha = 0.5)
        line, = axis2.plot([], [], marker = '.', lw = 0) 
        
        def init(): 
            line.set_data([], [])
            return line,
           
        def animate2(i):
            x = x_values
            y = probability_density_V2[i]
            line.set_data(x, y)
              
            return line,
        
        anim2 = FuncAnimation(fig2, animate2, init_func = init,
                             frames = len(time_values), interval = 10, blit = True)
        
        anim2.save('Wave_V2_split' + str(v0) + '.gif', dpi=100, 
                  writer=PillowWriter(fps=30))
    return
    
# Calling the animation function
start_time = time.time()
animation_wave_scattering(V0)
time_split = time.time() - start_time

print('Time for Split Operator Method: ', time_split)






