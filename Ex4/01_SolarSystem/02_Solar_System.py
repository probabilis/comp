# 02 - Simulation of the solar system

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

"""
Tasks:
* Determining the orbits of several n body systems
* Plotting orbitals of the planets and the sun
* Plotting the distance r(t) of the planets and sun to the center of mass
* Check for conversation of energy
* Check for conversation of angular momentum
* Check for agreement with orbital periods from literature
"""


import numpy as np
import matplotlib.pyplot as plt
from runge_kutta import explicit_runge_kutta
from skyfield.api import load
from scipy.fft import fft, fftfreq

# rk4-coefficients for Runge Kutta

a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])


# Masses of sun and planets in order sun to neptune
mj = np.array([1.988500e+30, 3.302e+23, 4.8685e+24, 5.972e+24, 6.4171e+23,
               1.89818722e+27, 5.6834e+26, 8.6813e+25, 1.02409e+26])


# Starting conditions from skyfield package (https://rhodesmill.org/skyfield/)
ts = load.timescale()
eph = load('de421.bsp')
# Starting time in UTC (year, month, day, hour, minute, ...)
t = ts.utc(2022, 1, 7, 12, 0) 


# The time unit we use in this program is years and not seconds
seconds_per_year = 365*24*60*60


# Loading ephemeride data of all planets + sun 
sun = eph['sun']
mercury = eph['mercury barycenter']
venus = eph['venus barycenter']
earth = eph['earth barycenter']
mars = eph['mars barycenter']
jupiter = eph['jupiter barycenter']
saturn = eph['saturn barycenter']
uranus = eph['uranus barycenter']
neptune = eph['neptune barycenter']


# Systems that will examined
system_1 = np.array([0, 1, 2, 3, 4])
objects_1 = [sun, mercury, venus, earth, mars]

system_2 = np.array([0, 5, 6, 7, 8])
objects_2 = [sun, jupiter, saturn, uranus, neptune]

system_3 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
objects_3 = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]


def solar_system(system, objects, t , t0, tmax, epsilon):
    
    """
    Input parameters:
    
    * system  ... numbers of the objects in the solar system (sun = 0) as array
    * objects ... ephemerides of the objects in the solar system
    * t       ... UTC starting time
    * t0      ... should be 0
    * tmax    ... maximum time in years
    
    Output:
    
    * Orbitals of the objects as a 3d plot
    * Distance r(t) of the objects against time in a plot
    * Orbital periods in years --> in above plot
    * Plots that show: Conservation of energy, angular momentum
    """
    
    # Names of the objects for labels in plot
    sol_sys = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn',
               'uranus', 'neptune']
    
    
    # Getting the initial conditions with skyfield package
    initial_conditions = []

    for obj in objects:
        position = obj.at(t).position.m
        velocity = obj.at(t).velocity.m_per_s
        initial_conditions.append(position[0])
        initial_conditions.append(position[1])
        initial_conditions.append(position[2])
        initial_conditions.append(velocity[0]*seconds_per_year)
        initial_conditions.append(velocity[1]*seconds_per_year)
        initial_conditions.append(velocity[2]*seconds_per_year)  

    y0 = np.array(initial_conditions)
    
    N_objects = len(system)
    # Gravitational constant in m^3/(kg*year^2)
    G_year = 6.6743*10**(-11)*seconds_per_year**2
    # Only the masses of the chosen system are in mj_red
    mj_red = mj[system]
    
    
    def solar_system_de(f, t):
        
        """
        Function that returns the system of first order differential equations:
        
        d(x1, y1, z1, vx1, vy1, vz1, ...) = (vx1, vy1, vz1, ax1, ay1, az1, ...)
        """
        
        # Initalizing size of df array
        df = np.zeros([N_objects*6,], dtype = np.double)

        # Indices for the f array 
        xi  = np.arange(0, N_objects*6, 6)
        yi  = np.arange(1, N_objects*6, 6)
        zi  = np.arange(2, N_objects*6, 6)
        vxi = np.arange(3, N_objects*6, 6)
        vyi = np.arange(4, N_objects*6, 6)
        vzi = np.arange(5, N_objects*6, 6)
    
        # Indices for the df array
        dvxi = xi
        dvyi = yi
        dvzi = zi
        daxi = vxi
        dayi = vyi
        dazi = vzi
    
        # the velocites of f can already be assigned to df
        df[dvxi] = f[vxi]
        df[dvyi] = f[vyi]
        df[dvzi] = f[vzi]
        
        # Initializing empty arrays for the acceleration components
        ax = np.zeros(N_objects,)
        ay = np.zeros(N_objects,)
        az = np.zeros(N_objects,)
    
        
        # Calculating the accelerations for every object
        for a in range(N_objects):
            
            # List of G*mj without G*ma 
            Gm = G_year*np.delete(mj_red, a, 0)
            # Positions of object a
            xa = f[xi][a]
            ya = f[yi][a]
            za = f[zi][a]
        
            # Calculating the distances r_ja
            xlist = np.delete(f[xi] - xa, a, 0)
            ylist = np.delete(f[yi] - ya, a, 0)
            zlist = np.delete(f[zi] - za, a, 0)
            
            rja3 = np.sqrt(np.square(xlist) + np.square(ylist) + np.square(zlist))**3
            # This is only for the first step in Runge Kutta to avoid an error
            if np.sum(rja3) == 0:
                ax[a] = 0
                ay[a] = 0
                az[a] = 0
           
            # Calculating the accelerations
            else:
                ax[a] = np.sum(Gm*xlist/rja3)
                ay[a] = np.sum(Gm*ylist/rja3)
                az[a] = np.sum(Gm*zlist/rja3)
        
        # assigning the accelerations to the df array
        df[daxi] = ax
        df[dayi] = ay
        df[dazi] = az
    
        return df
    
    
    # Calling Runge Kutta with rk4-coefficients
    y, t_plot = explicit_runge_kutta(solar_system_de, y0, t0, tmax, epsilon, a_rk4, b_rk4, c_rk4)
    
    
    
    # 3d plot of the planet + sun orbits around center of mass
    ax = plt.axes(projection='3d')
    
    for i, j in zip(range(len(system)), system):
        xdata = y[int(0 + i*6)]
        ydata = y[int(1 + i*6)]
        zdata = y[int(2 + i*6)]
        ax.scatter3D(xdata, ydata, zdata, marker = '.', label = sol_sys[j])
        
    ax.legend()
    
    
    # Plot of distance r(t) from center of mass + calculation of orbital period
    plt.figure(figsize = (20,8))
    
    for i, j in zip(range(N_objects), system):
        xdata = y[int(0 + i*6)]
        ydata = y[int(1 + i*6)]
        zdata = y[int(2 + i*6)]
        r_t = np.sqrt(xdata**2 + ydata**2 + zdata**2)
        
        # FFT to calculate frequence and orbital period
        N = len(t_plot)
        yf = fft(r_t/(1e+12))[:N//2]
        xf = fftfreq(N, epsilon)[0:N//2]
        yf_max = np.argmin(yf)
        orbital_period = 1 / xf[yf_max]
        
        plt.plot(t_plot, r_t, linestyle = '-', label = sol_sys[j] + ': $T = %.3f$ years'
                 %orbital_period)
        plt.title('Distance of the planets + sun from center of mass', fontsize = 20)
        plt.xlabel('$t$ / year')
        plt.ylabel('$r(t)$ / m')
        plt.legend()
    
    
    # Initializing arrays for |L(t)|, T(t), U(t)
    angular_momentum = np.zeros((len(t_plot)))
    kinetic_energy = np.zeros((len(t_plot)))
    potential_energy = np.zeros((len(t_plot)))
    
    # The x,y,z coordinates of all bodys are needed for U
    xi  = np.arange(0, N_objects*6, 6)
    yi  = np.arange(1, N_objects*6, 6)
    zi  = np.arange(2, N_objects*6, 6)
    all_x = y[xi]
    all_y = y[yi]
    all_z = y[zi]
    
    for i, m in zip(range(N_objects), mj_red):
        
        # Getting the x,y,z, vx,vy,vz data for body i
        xdata = y[int(0 + i*6)]
        ydata = y[int(1 + i*6)]
        zdata = y[int(2 + i*6)]
        vxdata = y[int(3 + i*6)]
        vydata = y[int(4 + i*6)]
        vzdata = y[int(5 + i*6)]
        
        # Calculating G*mj*mi for the potential
        GmM = G_year*m*np.delete(mj_red, i, 0)
        
        for t in range(len(t_plot)):
            
            # Vectors needed for L and T
            r_vector = np.array([xdata[t], ydata[t], zdata[t]])
            v_vector = np.array([vxdata[t], vydata[t], vzdata[t]])
            
            # L = sum(mi* ri x vi)
            L = m*np.cross(r_vector, v_vector)
            angular_momentum[t] += np.linalg.norm(L)
            
            # T = sum(mi * vi**2/2)
            T = m*np.linalg.norm(v_vector)**2/2
            kinetic_energy[t] += T
            
            # Calculating xj - xi, yj - yi, zj - zi
            xij = np.delete(all_x[:,t] - xdata[t], i, 0)
            yij = np.delete(all_y[:,t] - ydata[t], i, 0)
            zij = np.delete(all_z[:,t] - zdata[t], i, 0)
            
            rij = np.sqrt(np.square(xij) + np.square(yij) + np.square(zij))
            U = -np.sum(GmM/rij)
            potential_energy[t] += U
            
    
    # Plotting the norm of L against time
    plt.figure(figsize = (20,8))
    plt.plot(t_plot, angular_momentum)
    #plt.ylim((2.8e+48, 3e+48))          # system 1
    #plt.ylim((9.7e+50, 10e+50))         # system 2
    #plt.ylim((9.87e+50, 9.89e+50))      # system 3
    plt.title('Conservation of angular momentum over time', fontsize = 20)
    plt.grid(alpha = 0.5)
    plt.xlabel('$t$ / years')
    plt.ylabel('$|\\vec{L}|$ / kg$\cdot$m$^2\cdot$year$^{-1}$')
    
  
    # Plotting the absolute value of the energy E = T + U
    plt.figure(figsize = (20,8))
    plt.plot(t_plot, np.absolute(kinetic_energy + potential_energy/2))
    plt.title('Conservation of energy over time', fontsize = 20)
    
    #plt.ylim((5.5e+48, 6.5e+48))         # system 1
    #plt.ylim((1.90e+50, 1.91e+50))       # system 2
    #plt.ylim((1.96e+50, 1.98e+50))       # system 3
    plt.grid(alpha = 0.5)
    plt.xlabel('$t$ / years')
    plt.ylabel('$|E|$ / kg$\cdot$m$^2\cdot$year$^{-2}$')


#solar_system(system_1, objects_1, t , t0 = 0, tmax = 40, epsilon = 0.0002)  
#solar_system(system_1, objects_1, t , t0 = 0, tmax = 1, epsilon = 0.001) 
#solar_system(system_2, objects_2, t , t0 = 0, tmax = 165, epsilon = 0.01)  
#solar_system(system_2, objects_2, t , t0 = 0, tmax = 10000, epsilon = 0.05)
#solar_system(system_3, objects_3, t , t0 = 0, tmax = 30, epsilon = 0.01)


"""
From Wikipedia:
    
Mercury = 0.241 years
Venus = 0.616 years
Earth = 1.000 years
Mars = 1.882 years
Jupiter = 11.862 years
Saturn = 29.457 years
Uranus = 84.021 years
Neptune = 164.8 years
"""





