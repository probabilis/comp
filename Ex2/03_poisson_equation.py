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

# c + g) Solving for the electrostatic potential of a dipol configuration

#rho = np.zeros(N**2)

#rho[2550] = 50
#rho[7550] = -50
#h = 0.1

#phi_sol = gauss_seidel(A_laplace, -h**2*rho, 100, 1e-8)

#x_list = []

#y_list = []

#for k in range(N**2): 
    #x = k % N
    #y = (k - x)/N
    #x_list.append(x)
    #y_list.append(y)
    

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi = 200)
#surf = ax.plot_trisurf(x_list, y_list, phi_sol, cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)
#ax.set_xlabel('$x$')
#ax.set_ylabel('$y$')
#ax.set_zlabel('$\\phi(x,y)$')
#plt.show()


#############################################
#############################################
#############################################

# e) Speeding up the Gauss-Seidel alghorithm


A = discretized_laplace2d(4)

s = np.zeros(N**2, dtype = object)

for i in range(len(A)):
    s[i] = np.where(A[i] == 1)[0]
    
print(s)

print(A[0][s[0]])
    

    
    












