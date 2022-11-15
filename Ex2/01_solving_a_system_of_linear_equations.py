# 01 - Solving a system of linear equations

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

import numpy as np
import matplotlib.pyplot as plt


# a1) Implementing the Gauss-Seidel method


def gauss_seidel(A, b, limit, tol):
    
    x = np.zeros_like(b)
    u = 0
    n = len(A)
    
    for k in range(n):
        if (np.sum(np.abs(A[k, :k])) + np.sum(np.abs(A[k, k+1:]))) < np.abs(A[k,k]):
            u += 1
    
    if u == n:
        print('The matrix is strictly diagonally dominant!')
    
    else:
        print('The matrix is NOT strictly diagonally dominant!')
     
    x_sol = []
    
    for lim in range(limit):
        x_sol.append(x.copy())
        x_new = np.zeros_like(x)
        
        for i in range(n):
            sum_1 = np.sum(A[i, :i]*x_new[:i])
            sum_2 = np.sum(A[i, i + 1 :]*x[i + 1 :])
            x_new[i] = (b[i] - sum_1 - sum_2) / A[i,i]
            
        if np.allclose(x, x_new, rtol= tol):
            break
        
        if lim == (limit-1):
            print('Procedure does not converge! Set your limit higher.')
            
        x = x_new
        
    return np.array(x_sol)


#############################################
#############################################
#############################################

# a2) Implementing the Jacobi method


def jacobi(A, b, limit, tol):
    
    x = np.zeros_like(b)
    u = 0
    n = len(A)
    
    for k in range(n):
        if (np.sum(abs(A[k, :k])) + np.sum(abs(A[k, k+1:]))) < abs(A[k,k]):
            u += 1
            
    if u == n:
        print('The matrix is strictly diagonally dominant!')
    
    else:
        print('The matrix is NOT strictly diagonally dominant!')
         
    x_sol = []
    for lim in range(limit):
        x_sol.append(x.copy())
        x_new = np.zeros_like(x)
        for i in range(n):
            summe = 0
            for j in range(n):
                if i == j:
                    summe += 0
                else:
                    summe += A[i, j] * x[j]
            x_new[i] = (b[i] - summe)/A[i,i]
            if x_new[i] == x_new[i-1]:
                break
        if np.allclose(x, x_new, atol= tol, rtol= 0):
            break
        if lim == (limit-1): 
            print('Procedure does not converge! Set your limit higher.')
        x = x_new
    return np.array(x_sol)


#############################################
#############################################
#############################################

# b) Solving a random system of linear equations

# Defining a random NxN matrix which satisfies the convergence criteria

N = 100
A = np.random.uniform(0,1,(N,N))

# Replacing the diagonal elements to satisfy |a_ii| > sum(|a_ij|)
for i in range(len(A)):
    A[i,i] = np.random.uniform(N,N + 50)
    
b = np.random.uniform(0,N, size = N)
limit = 100
tol = 1e-8

x_sol_gauss = gauss_seidel(A, b, limit, tol)
x_sol_jacobi = jacobi(A, b, limit, tol)

error_gauss = []
error_jacobi = []

for p in range(len(x_sol_gauss)):
    e = np.sum(np.abs(np.dot(A, x_sol_gauss[p]) - b))
    error_gauss.append(e)

for p in range(len(x_sol_jacobi)):
    e = np.sum(np.abs(np.dot(A, x_sol_jacobi[p]) - b))
    error_jacobi.append(e)
    
iterations_gauss = np.arange(1,len(x_sol_gauss) + 1, 1)
iterations_jacobi = np.arange(1,len(x_sol_jacobi) + 1, 1)

# Plotting the difference of |b - b(p)| over number of iterations
plt.figure(figsize = (10,7), dpi = 300)
plt.plot(iterations_gauss, error_gauss, color = 'royalblue', linewidth = 0.7, marker = 'o', label = 'Gauss-Seidel')
plt.plot(iterations_jacobi, error_jacobi, color = 'darkgreen', linewidth = 0.7, marker = 'x', label = 'Jacobi')
plt.grid(alpha = 0.5)
plt.ylabel('$\sum_{i = 1}^{N} |b_i - b^{(p)}_i|$', fontsize = 12)
plt.xlabel('Iterations $p$', fontsize = 12)
plt.title('Comparison of Gauss-Seidel and Jacobi Method', fontsize = 16)
plt.legend(fontsize = 12)
plt.plot()

