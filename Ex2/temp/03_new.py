# 03 - Poisson equation in more than one dimension

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from gauss_seidel_float import gauss_seidel
import time

# a) implementing a function that returns all neighbours in d dimensions

# For simplicity reasons, we demand N1 =...= Nj = N points in each dimension

def nb(k, N, d):
    product = 1
    neighbours_k = []
    k_n = k
    
    n = np.zeros(d)
    
    # finding the coefficients n_i to find neighbours
    for i in range(d):
        n[i] = k_n % N
        k_n = k_n- n[i]
        k_n = k_n/N
    
    # if a coefficient n_i is 0 or N-1, it is on the domain in the i-th dimension
    for i in range(d):
        n_plus = k + product
        n_minus = k - product
        if n[i] == 0:
            neighbours_k.append(n_plus)
        if n[i] == (N-1):
            neighbours_k.append(n_minus)
        if (n[i] > 0) and (n[i] < N-1):
            neighbours_k.append(n_plus)
            neighbours_k.append(n_minus)
        
        product = product*N
        
    return neighbours_k


#############################################
#############################################
#############################################

# b) Creating a discretized Laplace operator for a NxN grid with periodic 
#    boundary conditions


def discretized_laplace2d(N):
    
    d = 2
    A = np.zeros((N**2, N**2))
    I = np.identity(N**2)
    A = A - 4*I
    
    # neighbours found by the nb function
    for k in range(len(A)):
        for i in nb(k, N, d):
            A[k][i] = 1
    
    # additional neighbours due to the periodic boundary conditions
    for i in range(N):
        A[i][N**2 - N + i] = 1
        A[N**2 - N + i][i] = 1
    
    for i in range(N):
        A[i*N][i*N + N-1] = 1
        A[i*N + N-1][i*N] = 1
    
    return A



N = 100
A_laplace = discretized_laplace2d(N)


#############################################
#############################################
#############################################
"""
# c + g) Solving for the electrostatic potential of a dipol configuration

rho = np.zeros(N**2)

rho[2550] = 50
rho[7550] = -50
h = 0.1

phi_sol = gauss_seidel(A_laplace, -h**2*rho, 100, 1e-8)

x_list = []

y_list = []

for k in range(N**2): 
    x = k % N
    y = (k - x)/N
    x_list.append(x)
    y_list.append(y)
    

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi = 200)
surf = ax.plot_trisurf(x_list, y_list, phi_sol, cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$\\phi(x,y)$')
plt.show()
"""
#############################################
#############################################
#############################################

# d) Speed Comparison

limit = 100
tol = 1e-8


##############################################################
#speed comparison

def speed_comparison(thres,limit,tol):
    #calcualtion to the N-th (threshold as integer) linear equation systems
    calc_times = np.zeros(thres-1)
    for N in range(2,thres+1):
        A = discretized_laplace2d(N)
        for i in range(len(A)):
            A[i,i] += 10
        b = np.zeros(N**2)
        b[:] = 50

        start_time_og = time.time()
        gauss_seidel(A,b,limit,tol)
        time_og = time.time() - start_time_og

        calc_times[N-2] = time_og

    return calc_times

#number of LES
thres = 20

calc_times = speed_comparison(thres, limit, tol)
range_n = np.arange(2,thres+1)
#print(calc_times)

y = lambda x : x**2 / 3000

plt.plot(range_n, calc_times[:], 'o', label = 'Gauss-Seidel')
plt.plot(range_n, y(range_n), label = '$n^2$')
plt.xlabel('$n$ points per dimension / 1')
plt.ylabel('time t / s')
plt.title('Calculation Time of Gauss-Seidel Method to solve Laplace equation')
plt.legend()





# e) Speeding up the Gauss-Seidel alghorithm

s = np.zeros(N**2, dtype = object)

for i in range(len(A_laplace)):
    s[i] = np.where(A_laplace[i] == 1)[0]


def adapted_gauss_seidel(A, b, s, limit, tol):
    
    x = np.zeros_like(b)
    n = len(A)

    x_sol = []

    for lim in range(limit):
        x_sol.append(x.copy())
        x_new = np.zeros_like(x)
        
        for i in range(n):

            indices_left = np.array([k for k in s[i] if k <= i])
            if len(indices_left) == 0:
                indices_left = False
        
            sum_1 = np.sum(A[i][indices_left]*x_new[indices_left])


            indices_right = np.array([k for k in s[i] if k > i])
            if len(indices_right) == 0:
                indices_right = False

            sum_2 = np.sum(A[i][indices_right]*x[indices_right])

            x_new[i] = (b[i] - sum_1 - sum_2) / A[i,i]

        if np.allclose(x, x_new, rtol= tol):
            break
        if lim == (limit-1):
            print('Procedure does not converge! Set your limit higher.')
        x = x_new
    return x_sol[-1]




rho = np.zeros(N**2)
rho[2550] = 50
rho[7550] = -50
h = 0.1





start_time_og = time.time()
phi_sol = gauss_seidel(A_laplace, -h**2*rho, limit, tol)
time_og = time.time() - start_time_og

start_time_adapt = time.time()
phi_sol_adapted = adapted_gauss_seidel(A_laplace, -h**2*rho, s, limit, tol)
time_adapt = time.time() - start_time_adapt

print('time_og', time_og)
print('time_adapt', time_adapt)
err = np.linalg.norm(phi_sol-phi_sol_adapted)
print(err)