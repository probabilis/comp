# 02 - Electron transmission through a molecular transport system

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

import numpy as np
import matplotlib.pyplot as plt
from gauss_seidel import gauss_seidel

# b) ring function to return the complex matrix (H_R - EI + i*Delta)

def ring(N, t, E, delta, alpha, beta):
    
    I = np.identity(N)
    
    Delta = np.zeros((N,N))
    i = complex(0,1)
    Delta[alpha-1, alpha-1] = delta
    Delta[beta-1, beta-1] = delta
    
    H_R = np.zeros((N,N))
    
    for a in range(N):
        for b in range(N):
            if (a == (b + 1)) or (a == (b - 1)) or (a == 0 and b == N-1) or (a == N-1 and b == 0):
                creator_destroyer = np.zeros((N,N))
                creator_destroyer[a,b] = t
                H_R = H_R + creator_destroyer
    
    matrix = H_R - E*I + i*Delta
    
    return matrix, H_R


#############################################
#############################################
#############################################

# c) solving for G_R for two different systems in E = [-6, 6]

# Input parameters 

alpha = [1,1] 
beta = [3,4]
N = 6
t = -2.6
delta = 0.5

# The Gauss Seidel alghorithm does not work for E = [-5.22, 5.22].
# The matrix is not diagonally dominant for these values!

E_1 = -6
E_2 = -5.23
E_3 = 5.23
E_4 = 6
E_ran_minus = np.arange(E_1, E_2 + 0.01, 0.01)
E_ran_plus = np.arange(E_3, E_4 + 0.01, 0.01)
E_ran = np.concatenate((E_ran_minus, E_ran_plus), axis = None)

limit = 1000
tol = 1e-8


def system_calc(N,t,E_ran,delta,alpha,beta, limit, tol):
    
    ring_matrices = np.zeros((len(E_ran),len(alpha)), dtype = object)

    #calculation of the ring matrices for different E_i and alpha/beta
    for j, E in enumerate(E_ran):
        k = -1
        for a, b in zip(alpha,beta):
            k += 1
            ring_matrices[j][k] = ring(N, t, E, delta, a, b)[0]
            
            
    #calculation of green functions
    I = np.identity(N)

    G_Rab = np.zeros((len(E_ran),len(alpha), N, N), dtype = object)

    for E in range(len(E_ran)):
        for i in range(len(alpha)):
            for n in range(len(I)):

                g_i = gauss_seidel(ring_matrices[E][i], I[n], limit, tol)

                G_Rab[E][i][n] = g_i
            G_Rab[E][i] = (G_Rab[E][i]).T

    return G_Rab, ring_matrices


GR_matrix, ring_matrices = system_calc(N,t,E_ran,delta,alpha,beta, limit, tol)


#############################################
#############################################
#############################################

# d) Plotting the transmission probability T_alpha_beta

fig, axs = plt.subplots(2, 2, figsize = (16,16))

axs_1 = axs[0,0]
axs_2 = axs[0,1]
axs_3 = axs[1,0]
axs_4 = axs[1,1]

# Plotting solutions with Gauss_Seidel

k = -1
for gr in GR_matrix[:]:
    k += 1
    axs_1.plot(E_ran[k], abs(gr[0][0][2])**2, '.', color = 'salmon', label = '$\\alpha = 1, \\beta = 3$')
    axs_2.plot(E_ran[k], abs(gr[1][0][3])**2, '.', color = 'aqua',label =  '$\\alpha = 1, \\beta = 4$')

E_ran_new = np.arange(-6, 6 + 0.01, 0.01)
ring_matrices_new = np.zeros((len(E_ran_new),len(alpha)), dtype = object)

# Plotting solutions with numpy solver which can handle all E's

for j, E in enumerate(E_ran_new):
    k = -1
    for a, b in zip(alpha,beta):
        k += 1
        ring_matrices_new[j][k] = ring(N, t, E, delta, a, b)[0]

for i in range(len(E_ran_new)):
    g_i1 = np.linalg.inv(ring_matrices_new[i][0])
    g_i2 = np.linalg.inv(ring_matrices_new[i][1])
    axs_3.plot(E_ran_new[i], abs(g_i1[0][2])**2, '.', color = 'salmon')
    axs_4.plot(E_ran_new[i], abs(g_i2[0][3])**2, '.', color = 'aqua')

axs_1.set_title('Transmission probability $T_{13}(E)$ with Gauss_Seidel')
axs_2.set_title('Transmission probability $T_{14}(E)$ with Gauss_Seidel')
axs_3.set_title('Transmission probability $T_{13}(E)$ with Numpy-Solver')
axs_4.set_title('Transmission probability $T_{14}(E)$ with Numpy-Solver')

axs_3.set_xlabel('E')
axs_4.set_xlabel('E')

axs_1.set_ylabel('Transmission probability $T_{\\alpha\\beta}(E)$')
axs_3.set_ylabel('Transmission probability $T_{\\alpha\\beta}(E)$')


# Calculating the eigenvalues of H_R

H_R = ring(N, t, E, delta, alpha[0], beta[0])[1]
eigenvalues = np.linalg.eigvals(H_R)

print('Eigenwerte H_R = ', eigenvalues)












