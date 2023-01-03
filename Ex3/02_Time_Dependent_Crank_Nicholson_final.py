# 02 - The time-dependent Schrödinger equation with Crank-Nicolson

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import time

# a) Crank-Nicolson routine

# Important note: h-bar is set to 1 in this exercise

def crank_nicolson(Nx, dx, Nt, dt, psi0, V, V0, m):
    
    """
    input parameters:
        
    * Nx   ... number of spatial grid points
    * dx   ... distance between two adjacent grid points
    * Nt   ... maximum number of time steps
    * dt   ... size of time step
    * psi0 ... wave function at t = 0 as the initial condition
    * V    ... a function of the potential
    * V0   ... height of the potential steps 
    * m    ... parameter of the gaussian wave packet
    
    output paramters:
    
    * psi  ... wave function psi(x,t) as 2D-array, rows = time, columns = space 
    """
    
    i = complex(0,1)
    x_values = np.arange(-Nx*dx/3, 2*Nx*dx/3, dx)
    
    # Dimensions of the psi-output, Nt = number of steps -> Nt+1 points in time 
    x_dim = Nx
    t_dim = Nt + 1
    
    # Initialising the psi-array, cdouble to handle complex wave functions
    psi = np.zeros((t_dim, x_dim), dtype = np.cdouble)
    
    # 1) Initial and boundary conditions
    psi[0] = psi0(x_values)
    psi[0][0] = 0
    psi[0][Nx-1] = 0
    
    # 2) Calculating a-array (a0 = aN = 0, not needed but for simpler looping
    #    in same spatial dimension as psi). 
    # Important: N = Nx - 1 (0, 1, ..., N = N + 1 points)
    a = np.zeros(x_dim, dtype = np.cdouble)
    a[1] = 2*(1 + m*dx**2*V(x_values[1], V0) - i*2*m*dx**2/dt)
    
    for k in range(2, Nx - 1):
        term =  2*(1 + m*dx**2*V(x_values[k], V0) - i*2*m*dx**2/dt)
        a[k] = term - 1/a[k-1]
          
    # Initialsing omega and b arrays with same dimensions as psi
    # omega0, omegaN, b0, bN are set to zero but unused
    omega = np.zeros((t_dim, x_dim), dtype = np.cdouble)
    b = np.zeros((t_dim, x_dim), dtype = np.cdouble)
    
    # 3) Time loop 
    for n in range(0, Nt):
        
        # 4) Calculating the omega_nk 
        for k in range(1, Nx - 1):
            omega[n][k] = (- psi[n][k-1] + 2*(i*2*m*dx**2/dt + 1 + 
                           m*dx**2*V(x_values[k], V0))*psi[n][k] - psi[n][k+1])
    
        # 5) Calculating the b_nk
        b[n][1] = omega[n][1]
        
        for k in range(2, Nx - 1):
            b[n][k] = b[n][k-1]/a[k-1] + omega[n][k]
            
        # 6) Calculating the psi_nk with boundary condition = 0
        psi[n+1][0] = 0
        psi[n+1][Nx-1] = 0
        for k in range(Nx - 2, 0, -1):
            psi[n+1][k] = 1/a[k]*(psi[n+1][k+1] - b[n][k])
            
    return psi


#############################################
#############################################
#############################################

# b) Freely propagating Gaussian Wave Packet


def psi0_free(x, sigma = 10, x_0 = 0, q = 2, m = 1):
    
    """
    Gaussian wave packet at t = 0 with:
    * sigma = 10
    * x_0   = 0
    * q     = 2
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


#############################################
#############################################
#############################################

# c) Plotting the probability density of the free particle and checking the 
#    conservation of the total probability


# Parameters for the Crank-Nicolson Alghorithm
Nx = 8000
dx = 0.1
Nt = 1500
dt = 0.1
m = 1
V0_free = 0

# Calculating psi(x,t) of the free particle ny calling Crank-Nicolson
psi_free = crank_nicolson(Nx, dx, Nt, dt, psi0_free, V_free, V0_free, m)

# Calculating the probability density |psi(x,t)|^2
probability_density_free = np.absolute(psi_free)**2


# Checking for conservation of total probability
total_probability = []

for i in range(0,Nt+1):
    total_probability.append(np.sum(probability_density_free[i]*dx))
    
# (x,t) for plotting   
x_values = np.arange(-Nx*dx/3, 2*Nx*dx/3, dx)
time_values = np.arange(0, Nt*dt + dt, dt)

# Plotting the total probability over time

plt.figure(figsize = (10,7), dpi = 300)
plt.plot(time_values, total_probability, color = 'royalblue', linewidth = 1)
plt.grid(alpha = 0.5)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Total Probability', fontsize = 12)
plt.ylim((0.9, 1.1))
plt.title('Conservation of Total Probability using Crank-Nicolson', fontsize = 16)


# Animation of the propagating wave packet with matplotlib

fig = plt.figure(figsize = (10,7), dpi = 300) 
axis = plt.axes(xlim =(x_values[0], x_values[-1]), ylim =(-0.01, 0.06))
plt.title('Propagating Probability Density with $V_0(x) = 0$', fontsize = 16)
plt.xlabel('x', fontsize = 14)
plt.ylabel('$|\psi(x,t)|^2$', fontsize = 14)
plt.grid(alpha = 0.5)
line, = axis.plot([], [], lw = 3) 

def init(): 
    line.set_data([], [])
    return line,
   
def animate(i):
    x = x_values
    y = probability_density_free[i]
    line.set_data(x, y)
      
    return line,

anim = FuncAnimation(fig, animate, init_func = init,
                     frames = len(time_values), interval = 10, blit = True)

anim.save("Free_Wave.gif", dpi=100, 
          writer=PillowWriter(fps=30))


#############################################
#############################################
#############################################

# d + e) Scattering of the Gaussian wave packet

# Parameters of the potential barriers 
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
        psi_V1 = crank_nicolson(Nx, dx, Nt, dt, psi0_scatter, V_1, v0, m)
        psi_V2 = crank_nicolson(Nx, dx, Nt, dt, psi0_scatter, V_2, v0, m)
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
        
        anim1.save('Wave_V1_' + str(v0) + '.gif', dpi=100, 
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
        
        anim2.save('Wave_V2_' + str(v0) + '.gif', dpi=100, 
                  writer=PillowWriter(fps=30))
    return
    

# Calling the animation function
start_time = time.time()
animation_wave_scattering(V0)
time_split = time.time() - start_time

print('Time for Crank-Nicolson: ', time_split)


        