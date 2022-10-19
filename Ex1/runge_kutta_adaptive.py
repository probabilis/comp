#Runge Kutta Adaptive Time Stepping with Double Stepping

import numpy as np

import matplotlib.pyplot as plt
from runge_kutta import explicit_runge_kutta

import time

#RK4 method
a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])



def explicit_runge_kutta_adaptive(F, y0, t0, t_max, epsilon, a, b, c):
    """
    input parameters:

    F(y,t) = array of 1 order d.e. functions
    y0 = initial conditions in y
    t0 = initial condition in t
    t_max = maximal time value 
    epsilon = incremental size of differentation
    a, b, c = coefficients for specific method (RK4, Euler,.. )

    output:
    y, t = solution vector  in y and time vector t

    """
    nr_y = y0.size
    
    #t = np.arange(t0, t[n], epsilon)
    #nr_t = t.size

    y_tol = 0.01
    eps_min = 0.001

    dy_min = 0.00001
    dy_max = 0.01

    t_vec_max = round((t_max - t0) / eps_min) * 3 
    
    y = np.zeros((nr_y, t_vec_max))
    y[:,0] = y0

    y_half = np.copy(y)
    y_double = np.copy(y)
    
    d = c.size
    k = np.zeros((nr_y, t_vec_max, d))
    k_half = np.copy(k)
    k_double = np.copy(k)

    t = t0
    n = -1
    t_list = []
    
    while t < t_max:
        n = n + 1
        for i in range(0,d):

            delta_k = 0 ; delta_k_half = 0 ; delta_k_double = 0

            delta_k = np.sum(a[i,:] * F( k[:,n,:], t + c[:] * epsilon ), axis = 1)
            delta_k_half = np.sum(a[i,:] * F( k_half[:,n,:], t + c[:] * epsilon * 0.5), axis = 1)
            delta_k_double = np.sum(a[i,:] * F( k_double[:,n,:], t + c[:] * epsilon * 2), axis = 1)
                
            k[:,n,i] = y[:,n] + epsilon * delta_k
            k_half[:,n,i] = y[:,n] + 0.5 * epsilon * delta_k_half
            k_double[:,n,i] = y[:,n] + 2 * epsilon * delta_k_double
            
            delta_y = 0 ; delta_y_half = 0 ; delta_y_double = 0

            delta_y = np.sum(b[:]*F(k[:,n,:], t + epsilon * c[:]), axis = 1)
            delta_y_half = np.sum(b[:]*F(k_half[:,n,:], t + 0.5 * epsilon * c[:] ), axis = 1)
            delta_y_double = np.sum(b[:]*F(k_double[:,n,:], t + 2 * epsilon * c[:] ), axis = 1)   


        y[:,n+1] = y[:,n] + epsilon * delta_y
        y_half[:,n+1] = y[:,n] + 0.5 * epsilon * delta_y_half
        y_double[:,n+1] = y[:,n] + 2 * epsilon * delta_y_double

        print('Classic' , y[:,n+1])
        print('Half', y_half[:,n+1])
        print('Double', y_double[:,n+1])

        print('Epsilon', epsilon)

        #criteria for de- or increasing epsilon for the worst (= min./max.) value in each ODE system
        if (np.max(np.abs( y[:,n+1] )) < y_tol ):
            if (epsilon != eps_min):
                print("New step size 1", eps_min)
                epsilon = eps_min
            y_new = y[:,n+1]
        
        else:
            if ( np.min(np.abs( y[:,n+1] )) > y_tol and np.min(np.abs( y[:,n+1] - y_half[:,n+1] ) / np.abs(y[:,n+1] )) > dy_max):
                epsilon = epsilon / 2
                print("New step size 2 ", epsilon)
                y_new = y_half[:, n+1]
            elif ( np.min(np.abs( y[:, n+1] )) > y_tol  and np.max(np.abs(y[:,n+1] - y_double[:,n+1] ) / np.abs(y[:,n+1] )) < dy_min ):
                epsilon = 2 * epsilon
                print("New step size 3 ", epsilon)
                y_new = y_double[:, n+1]
            else:
                y_new = y[:,n+1]
                print('No adjustment')

        y[:,n+1] = y_new
        t_list.append(t)
        t = t + epsilon
    return y, t_list

#############################################

#1b
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
epsilon = 0.1


st = time.time()
y_rk4, t_rk4 = explicit_runge_kutta(ode_simple_pendulum, y0, t0, t_max, epsilon , a_rk4, b_rk4, c_rk4)
et = time.time()

st_a = time.time()
y_rk4_a, t_rk4_a = explicit_runge_kutta_adaptive(ode_simple_pendulum, y0, t0, t_max, epsilon, a_rk4, b_rk4, c_rk4)
et_a = time.time()


plt.plot(t_rk4, y_rk4[0,:], color = 'mediumseagreen', label = 'RK4')
#plt.plot(t_rk4, y_rk4[1,:], color = 'salmon', label = 'RK4')

plt.plot(t_rk4_a, y_rk4_a[0,0:len(t_rk4_a)], color = 'skyblue', label = 'RK4 adaptive')

plt.xlabel('$t$ / s')
plt.ylabel('$\\Theta(t)$')
plt.legend()
plt.title('Solving the pendulum differential equation')



elapsed_time = et - st
elapsed_time_a = et_a - st_a
print('Classic RK', elapsed_time, 'seconds')
print('Adaptive RK', elapsed_time_a, 'seconds')

