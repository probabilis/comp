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
        k_n = k_n - n[i]
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

# b) Creating the discretized Laplace operator for a NxN grid with periodic 
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

# Creating the NxN matrix by calling the function

N = 100
A_laplace = discretized_laplace2d(N)


#############################################
#############################################
#############################################

# c + g) Solving for the electrostatic potential of a dipol configuration
# and generating a 3D surface plot

# Setting two points of the grid to +- rho
rho = np.zeros(N**2)
rho[2550] = 50
rho[7550] = -50

h = 0.1

# Calling Gauss-Seidl to solve for the phi_k
phi_sol = gauss_seidel(A_laplace, -h**2*rho, 100, 1e-8)

# Generating the grid for plotting -> calculating (x,y) from k
x_list = []
y_list = []

for k in range(N**2): 
    x = k % N
    y = (k - x)/N
    x_list.append(x)
    y_list.append(y)
    
# 3D Surface Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi = 200)
surf = ax.plot_trisurf(x_list, y_list, phi_sol, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$\\phi(x,y)$')
plt.show()


#############################################
#############################################
#############################################

# d) Computational time with grid size -> quadratic behaviour noticed

limit = 100
tol = 1e-8


def speed_comparison(thres,limit,tol):
    
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

# maximum number of points per dimension for the speed comparison 
thres = 20

calc_times = speed_comparison(thres, limit, tol)
range_n = np.arange(2,thres+1)
#print(calc_times)

# Quadratic function for comparison to computational size
y = lambda x : x**2 / 3000

plt.figure(figsize = (8,8))
plt.plot(range_n, calc_times[:], 'o', label = 'Gauss-Seidel')
plt.plot(range_n, y(range_n), label = '$n^2$')
plt.xlabel('$n$ points per dimension / 1')
plt.ylabel('time t / s')
plt.title('Calculation Time of Gauss-Seidel Method to solve Laplace equation')
plt.legend()
plt.show()


#############################################
#############################################
#############################################


# e) Speeding up the Gauss-Seidel alghorithm

# Here we get the array with indices where the matrix element = 1.
# Important: The adapted Gauss-Seidel alghorithm with the generated array
# is indeed faster, but only if we have this array in advance. 
# If we generate the array s in Gauss-Seidel, it is slower than normal!

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

# Comparing the two Gauss-Seidel alghorithms for the Dipole problem

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
# Checking if the solutions are equal
err = np.linalg.norm(phi_sol-phi_sol_adapted)
print(err)



#############################################
#############################################
#############################################

# f) Faraday Cage

# Defining a function that returns True, if the point is on the rim of the cage
def faraday_cage(i, N):
    
    # bottom rim
    for k in range((N//4)*N + N//4, (N//4)*N + (N*3)//4):
        if i == k:
            return True
    # top rim    
    for k in range((N*3)//4 * N + N//4, (N*3)//4 * N + (N*3)//4):
        if i == k:
            return True
    # left rim   
    for k in [(N//4 + i)*N + N//4 for i in range(N//2)]:
        if i == k:
            return True
    # right rim
    for k in [(N//4 + i)*N + (N*3)//4 for i in range(N//2)]:
        if i == k:
            return True
        
# New neighbour function that returns left, right, top, bottom and periodic
# (right and left grid domain) neighbours but excludes neighbours from the 
# bottom and top line
def nb_faraday(k, N):
    d = 2
    product = 1
    neighbours_k = []
    k_n = k
    
    n = np.zeros(d)
    
    # finding the coefficients n_i to find neighbours
    for i in range(d):
        n[i] = k_n % N
        k_n = k_n - n[i]
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
    
    # Removing neighbours that are on the bottom or top domain
    for i in range(N):
        if i in neighbours_k:
            neighbours_k.remove(i)
    for i in range(N**2 - N, N**2):
        if i in neighbours_k:
            neighbours_k.remove(i)
    
    # Periodic boundary conditions for the left and right domain             
    if n[0] == 0:
        neighbours_k.append(k + N - 1)
    if n[0] == N-1:
        neighbours_k.append(k - N + 1)
        
    return np.array(neighbours_k)


# Returns a reduced matrix (because we already know phi(x,0) and phi(x, L_y))
def faraday_laplace(N):
    
    A = np.zeros((N**2 - 2*N, N**2 - 2*N))
    
    for i in range(N, N**2 - N):
        # For points on the faraday cage -> phi = 0
        if faraday_cage(i,N) == True:
            A[i-N, i-N] = 1
        # For all other points
        else:
            A[i-N, i-N] = -4
            A[i-N, nb_faraday(i, N) - N] = 1
    
    return A
            
N = 50
A_faraday = faraday_laplace(N)

# Defining the reduced rho vector 
rho = np.zeros(N**2 - 2*N)
# The already known potentials are added to the right side of the equation
rho[np.arange(0,N)] = -1
rho[np.arange(N**2 - 3*N, N**2 - 2*N)] = 1

# Solutions of phi for the bottom and top domain
phi_lower = np.zeros(N) + 1
phi_upper = np.zeros(N) - 1

# Solution for rectangle between 
phi_sol_rec = gauss_seidel(A_faraday, rho, 100, 1e-8)

# Generating the full solution vector
phi_sol = np.concatenate((phi_lower, phi_sol_rec, phi_upper))

# Generating the plotting grid
x_list = []
y_list = []

for k in range(N**2): 
    x = k % N
    y = (k - x)/N
    x_list.append(x)
    y_list.append(y)
    
# 3D surface plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi = 200)
surf = ax.plot_trisurf(x_list, y_list, phi_sol, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$\\phi(x,y)$')
plt.show()
