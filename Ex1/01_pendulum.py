##01 - A simple Pendulum
#General Runge-Kutta solver for linear-diff.eq systems

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


#coefficients for each numerical method S

#RK4 method
a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])

#Euler method
a_eul = np.array([[0]])

b_eul = np.array([1])

c_eul = np.array([0])



#explicit Runge Kutta method for solving systems of ordinary differential equations

def explicit_runge_kutta(F, y0, t0, t_max, epsilon, a, b, c):
    """
    input parameters:
    F(y,t) = array of 1 order d.e. functions
    y0 = initial condition in y
    t0 = initial condition in t
    t_max = maximal time of 
    epsilon = incremental size of differentation
    a, b, c = coefficients for specific method

    output:
    y, t = solution vector  in y and time vector t
    """
    
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

#1b
def ode_simple_pendulum(f, t):
    l = 1
    g = 9.81
    omega = np.sqrt(g/l)
    theta, p = f

    return np.array([p , -omega**2 * np.sin(theta)])


#initial conditions
y0 = np.array([np.pi/30, 0])
t0 = 0
t_max = 10
epsilon = 0.05

y_rk4, t_rk4 = explicit_runge_kutta(ode_simple_pendulum, y0, t0, t_max, epsilon, a_rk4, b_rk4, c_rk4)
y_eul, t_eul = explicit_runge_kutta(ode_simple_pendulum, y0, t0, t_max, epsilon, a_eul, b_eul, c_eul)

plt.plot(t_eul, y_eul[0,:], color = 'salmon', label = 'Euler')
plt.plot(t_rk4, y_rk4[0,:], color = 'mediumseagreen', label = 'RK4')
plt.xlabel('$t$ / s')
plt.ylabel('$\\Theta(t)$')
plt.legend()
plt.title('Solving the pendulum differential equation')




#############################################
#############################################
#############################################

#1c
#plots for system Energy E(t) and Theta(t) for 3 different values of eps for Euler and RK4

#epsillon list
eps_list = [0.01, 0.05, 0.1]

#define plotting layout
fig, axs = plt.subplots(2, len(eps_list), sharex = 'col', figsize = (8,10))

#differential equation same as before
#see ode_simple_pendulum(f, t):


#calling function for calculation and plotting
def comparison_erk_eul(eps_list):
    t0 = 0
    l = 1 ; g = 9.81 ; m = 1
    omega = np.sqrt(g/l)
    y0 = np.array([0, 2 * omega])
    t_max = 10

    for i, eps in enumerate(eps_list):
        y_rk4, t_rk4 = explicit_runge_kutta(ode_simple_pendulum, y0, t0, t_max, eps, a_rk4, b_rk4, c_rk4)
        y_eul, t_eul = explicit_runge_kutta(ode_simple_pendulum, y0, t0, t_max, eps, a_eul, b_eul, c_eul)

        E_t_rk4 = 0.5*m*l**2*y_rk4[1,:]**2 + m*g*l*(1 - np.cos(y_rk4[0,:]))
        E_t_eul = 0.5*m*l**2*y_eul[1,:]**2 + m*g*l*(1 - np.cos(y_eul[0,:]))

        axs_theta = axs[0,i] ; axs_energy = axs[1,i]
        axs_theta.plot(t_eul, y_eul[0,:], color = 'salmon', label = 'Euler')
        axs_theta.plot(t_rk4, y_rk4[0,:], color = 'mediumseagreen' , label = 'RK4')
        axs_theta.legend()
        
        axs_energy.plot(t_rk4, E_t_rk4, color = 'skyblue', label = 'RK4')
        axs_energy.plot(t_eul, E_t_eul, color = 'crimson', label = 'Euler') 
        axs_energy.legend() 
    return

comparison_erk_eul(eps_list)

for ax_top, ax_bot, i in zip(axs[0], axs[-1], eps_list):
    ax_top.set_title('$\\epsilon$={}'.format(i))
    ax_bot.set_xlabel('$t$ / s'.format(i))
axs[0,0].set_ylabel('$\\Theta(t)$')
axs[1,0].set_ylabel('E(t)')


fig.suptitle('Pendulum - System Energy E(t) and $\\Theta$(t) for different values of $\\epsilon$ for Euler and RK4')
fig.tight_layout()
#fig.savefig('pendulum.png')
plt.show


#############################################
#############################################
#############################################


#1d
#plot for Theta(t) with RK4-routine for a long time-vektor


from scipy.fft import fft, fftfreq
from scipy.signal import blackman



fig, axs = plt.subplots(2, figsize = (8,10))

t0 = 0
y0 = np.array([np.pi/3, 0])
t_max = 20
eps = 0.05

y_rk4, t_rk4 = explicit_runge_kutta(ode_simple_pendulum, y0, t0, t_max, eps, a_rk4, b_rk4, c_rk4)

N = len(t_rk4)
yf = fft(y_rk4[0,:])[:N//2]
xf = fftfreq(N, eps)[0:N//2]

#yf_max for calculation of frequency
yf_max = np.argmax(yf)
osc_fr = 1 / xf[yf_max]

#plotting
axs[0].plot(t_rk4, y_rk4[0,:], color = 'mediumpurple', label = 'RK4')

axs[0].set_xlabel('time $t$ / s')
axs[0].set_ylabel('$\\Theta$(t) / rad')


axs[1].plot(xf, 2.0/N * np.abs(yf), color = 'lightblue')

axs[1].set_xlabel('frequency $\\omega$ / $s^{-1}$')
axs[1].set_ylabel('| FFT |')
axs[1].text(3, 0.5, '$T$ = {:.3f}'.format(osc_fr) + '  $s$', fontsize = 10)


fig.suptitle('Resolving the swinging periodicity $\\omega$')
fig.tight_layout()
axs[0].legend(loc = 'upper right')

plt.show()

print('$T$ = {:.3f}'.format(osc_fr) + '  $s$')




    
