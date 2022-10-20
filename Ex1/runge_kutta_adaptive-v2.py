# -*- coding: utf-8 -*-

#4 - Runge Kutta Adaptive Time Stepping with Felhberg Method
#IMPORTANT NOTE:
#algorithm is not working as expected, due to diverging solution over time
#in addditional, the operating time is not as expected
#
#
#reference: 
#https://www.wikiwand.com/en/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method


# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

import numpy as np
import matplotlib.pyplot as plt
#importing the global General-Explicit RK algorithm from the project folder
from runge_kutta import explicit_runge_kutta 
import time

#coefficients for each numerical method S

#RK4 method
a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])

############################################

#Fehlberg method
b_rkf45 = np.array([[0,0,0,0,0,0], 
                    [1/4,0,0,0,0,0],
                    [3/32, 9/32,0,0,0,0],
                    [1932/2197, -7200/2197, 7296/2197,0,0,0],
                    [439/216, -8, 3680/513, -845/4104, 0,0],
                    [-8/27,2,-3544/2565, 1859/4104, -11/40,0]])

c_rkf45 = np.array([25/216, 0, 1408/2565, 2197/1404, -1/5, 0])

a_rkf45 = np.array([0,1/4,3/8,12/13,1,1/2])

r_rkf45 = np.array([1/360, 0, -128/4275, -2197/75240, 1/50, 2/55])


def runge_kutta_f45(F, y0, t0, t_max, b, c, a, r, epsilon, hmax, hmin):
    """
    input parameters:

    F(y,t) = array of 1 order d.e. functions
    y0 = initial conditions in y
    t0 = initial condition in t
    t_max = maximal time value 
    b,c,a,r = coefficients for RK-Fehlberg method
    epsilon = tolerance threshold
    #h = incremental size of differentation -> adapated
    hmax = maximum incremental size of differentation
    hmin = minimum incremental size of differentation

    output:
    y, t = solution vector  in y and time vector t

    """
    t = t0 
    d = c.size
    n = 0
    nr_y = len(y0)
    h_list = []
    #arrays are filled in dimension 2 (t-axis) with a fixed amount of zeros due to unknown discreization of the whole length
    k = np.zeros((nr_y, 10000, d))
    y = np.zeros((nr_y, 10000))
    y[:,0] = y0

    T = np.array([t0])
    
    #setting in the beginning h value to hmax predefined threshold
    h = hmax
    while t < t_max:
        #print(n) 
        if t + h > t_max:
            h = t_max - t

        for i in range(0, d):
            delta_k = 0
            delta_k = np.sum(b[i,:] * F( k[:,n, :], t + a[:] * h ), axis = 1) 

            k[:,n,i] = y[:,n] + h * delta_k
        
        #calculating truncation error
        te = np.abs(np.sum(k[:,n,:] * r[:], axis = 1)) / h

        #getting the maximum truncation error
        if len( np.shape(te) ) > 0:
            te = np.max(te)

        #give output due to right te
        if te <= epsilon:    
            t = t + h
            
            delta_y = 0
            delta_y = np.sum(c[:] * F(k[:,n,:], t + h * a[:]), axis = 1) 
            
            y[:,n+1] = y[:,n] + delta_y * h

            T = np.append(T, t)
            h_list.append(h)
            n = n + 1

        #update h due to current truncation error 
        h = 0.9 * h * (epsilon / te)**0.2

        if h > hmax:
            h = hmax
        elif h < hmin:
            raise RuntimeError("Error: Could not converge to the required tolerance %e with minimum stepsize  %e." % (epsilon,hmin))
            break
        
    
    return y, T, h_list


def ode_simple_pendulum(f, t):
    l = 1
    g = 9.81
    omega = np.sqrt(g/l)
    theta, p = f

    return np.array([p, -omega**2 * np.sin(theta)])


#initial conditions
y0 = np.array([np.pi/30, 0])
t0 = 0
t_max = 10
epsilon = 0.05

#calling classical RK4 method and stopping operating time

st_rk4 = time.time()
y_rk4, t_rk4 = explicit_runge_kutta(ode_simple_pendulum, y0, t0, t_max, epsilon , a_rk4, b_rk4, c_rk4)
et_rk4 = time.time()

#calling Fehlberg-RK4 method and stopping operating time

st_f45 = time.time()
y_f45, t_f45, h_list =runge_kutta_f45(ode_simple_pendulum, y0, t0, t_max, b_rkf45, c_rkf45, a_rkf45, r_rkf45, epsilon = 1e-6, hmax = 1e-1, hmin = 1e-16)
et_f45 = time.time()

#operating time calculation and ouput

elapsed_time_rk4 = et_rk4 - st_rk4
elapsed_time_f45 = et_f45 - st_f45
print('Classic RK', elapsed_time_rk4, 'seconds')
print('Embedded RK Fehlberg method', elapsed_time_f45, 'seconds')


plt.plot(t_rk4, y_rk4[0], color = 'mediumseagreen', label = 'RK4')
plt.plot(t_f45, y_f45[0,0:len(t_f45)], color = 'salmon', label = 'RK-Fehlberg')
plt.plot(t_f45[1:], h_list, label = 'h adjustment')

plt.xlabel('$t$ / s')
plt.ylabel('$\\Theta(t)$')
plt.legend()
plt.title('RK4 / RK-Fehlberg method comparison')
plt.text(0.5, -0.2, '$t_R$ = {:.3f}'.format(elapsed_time_rk4) + '  $s$', fontsize = 10)
plt.text(0.5, -0.3, '$t_F$ = {:.3f}'.format(elapsed_time_f45) + '  $s$', fontsize = 10)
#plt.savefig('rk4-f45_comparison.png')

