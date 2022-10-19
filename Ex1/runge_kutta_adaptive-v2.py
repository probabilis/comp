#Runge Kutta Function Modul for CP Project


import numpy as np
import matplotlib.pyplot as plt
#importing the global General-Explicit RK algorithm from the project folder
from runge_kutta import explicit_runge_kutta 

#coefficients for each numerical method S

#RK4 method
a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])

############################################


b_rkf45 = np.array([[0,0,0,0,0,0], 
                    [1/4,0,0,0,0,0],
                    [3/32, 9/32,0,0,0,0],
                    [1932/2197, -7200/2197, 7296/2197,0,0,0],
                    [439/216, -8, 3680/513, -845/4104, 0,0],
                    [-8/27,2,-3544/2565, 1859/4104, -11/40,0]])

c_rkf45 = np.array([25/216, 0, 1408/2565, 2197/1404, -1/5, 0])

a_rkf45 = np.array([0,1/4,3/8,12/13,1,1/2])

r_rkf45 = np.array([1/360, 0, -128/4275, -2197/75240, 1/50, 2/55])


def runge_kutta_f45(F, y0, t0, t_max, b, c, a, r):
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
    tol = 1e-6
    t = t0

    hmax= 1e-1 ; hmin=1e-16

    nr_y = len(y0)
    
    d = c.size

    k = np.zeros((nr_y, 10000, d))
    n = 0
    
    y = np.zeros((nr_y, 10000))
    y[:,0] = y0

    T = np.array([t0])
    Y = np.array([y0])

    epsilon = hmax
    
    while t < t_max:
         
        print(n)
        if t + epsilon > t_max:
            epsilon = t_max - t

        for i in range(0, d):
            delta_k = 0
            delta_k = np.sum(b[i,:] * F( k[:,n, :], t + a[:] * epsilon ), axis = 1) 

            k[:,n,i] = y[:,n] + epsilon * delta_k
        
        #calculating truncation error
        te = np.abs(np.sum(k[:,n,:] * r[:], axis = 1)) / epsilon

        print('Array TE', te)
        #getting the maximum truncation error
        if len( np.shape(te) ) > 0:
            te = np.max(te)

        print('Max',te)

        #give output due to right te
        if te <= tol:    
            t = t + epsilon
            
            delta_y = 0
            delta_y = np.sum(c[:] * F(k[:,n,:], t + epsilon * a[:]), axis = 1) 
            
            y[:,n+1] = y[:,n] + delta_y * epsilon

            T = np.append(T, t)
            Y = np.append(Y, y)
            n = n + 1

        #update te 
        epsilon = 0.9 * epsilon * (tol / te)**0.2

        if epsilon > hmax:
            epsilon = hmax
        elif epsilon < hmin:
            raise RuntimeError("Error: Could not converge to the required tolerance %e with minimum stepsize  %e." % (tol,hmin))
            break
    
    return y, T


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
#epsilon = 0.05

y_rk4, t_rk4 =runge_kutta_f45(ode_simple_pendulum, y0, t0, t_max, b_rkf45, c_rkf45, a_rkf45, r_rkf45)

#

plt.plot(t_rk4, y_rk4[0,0:len(t_rk4)], color = 'mediumseagreen', label = 'RK4')
plt.xlabel('$t$ / s')
plt.ylabel('$\\Theta(t)$')
plt.legend()
plt.title('Solving the pendulum differential equation')