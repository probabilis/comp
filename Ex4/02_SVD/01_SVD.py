# 01 - Singular value decomposition of a drawing

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Loading the image as a np.array --> dimensions (height, length, rgb)

im = np.asarray(Image.open('Codice_Atlantico.bmp'))

# Matrices A_nu with dimensions (height, length) containing the intensities
# of the colors red, green and blue.
# Data Type np.float64 is essential, otherwise it doesn't work
A_r = im[:,:,0].astype(np.float64)
A_g = im[:,:,1].astype(np.float64)
A_b = im[:,:,2].astype(np.float64)

# Calculating B = A_conj*A
B_r = np.dot(np.conjugate(np.transpose(A_r)), A_r)
B_g = np.dot(np.conjugate(np.transpose(A_g)), A_g)
B_b = np.dot(np.conjugate(np.transpose(A_b)), A_b)


#################################    
#################################
#################################

# a) Implementing the power method


# Rayleigh Quotient, which is used to calculate eigenvalues in the power method
def rayleigh_quotient(A,x):
    
    numerator = np.dot(np.conjugate(x), np.dot(A,x))
    denominator = np.dot(np.conjugate(x), x)
    lam = numerator/denominator
    
    return lam



def power_method(B, tol, maxit, x_0):
    """

    Parameters
    ----------
    B     ... input matrix
    tol   ... possible value 1e-5
    maxit ... maximum number of iterations
    x_0   ... starting vector, may be set to None

    Returns
    -------
    eigenvalue  ... largest eigenvalue of matrix B
    eigenvector ... corresponding normalized eigenvector
    residuum    ... list of |lam(p) - lam(p-1)|

    """
    
    # Setting x to the input value x_0 --> random vector if x_0 = None
    if x_0 == None:
        eigenvector = np.random.rand(B.shape[1])
    else:
        eigenvector = x_0
    
    # Calculating the first eigenvalue with the Rayleigh-Quotient
    eigenvalue = rayleigh_quotient(B,eigenvector)
    
    residuum = []
    # Number of iterations 
    p = 0
    
    # Implementing the power method
    while p <= maxit:
        
        # New eigenvector
        eigenvector = np.dot(B,eigenvector)
        eigenvector = eigenvector/np.linalg.norm(eigenvector)
        
        # New eigenvalue with Rayleigh-Quotient
        eigenvalue_new = rayleigh_quotient(B, eigenvector)
        
        # Calculating the difference with the eigenvalue of the step before
        r_p = abs(eigenvalue_new - eigenvalue)
        residuum.append(r_p)
        
        # Updating for the next step
        eigenvalue = eigenvalue_new
        p += 1
        
        # The alghorithm breaks if the difference is below the tolerance
        if r_p <= tol:
            return eigenvalue, eigenvector, residuum
            break
        
        if p == maxit and r_p > tol:
            print('Not enough iterations to reach desired tolerance')
            return eigenvalue, eigenvector, residuum
        

#################################    
#################################
#################################

# b) Calculating the n largest eigenvalues using power method and deflation


def largest_eigvals(B, n, tol, maxit, x_0):
    """

    Parameters
    ----------
    B     ... input matrix
    tol   ... possible value 1e-5
    maxit ... maximum number of iterations
    x_0   ... starting vector, may be set to None
    n     ... number of eigenvalues that will be calculated

    Returns
    -------
    eigenvalues  ... list of the n largest eigenvalues of matrix B
    eigenvectors ... list corresponding normalized eigenvectors
    residuums    ... list of residuums for every eigenvalue

    """
    
    eigenvalues = []
    eigenvectors = []
    residuums = []
    
    for i in range(n):
        
        # Calculating the largest eigenvalue for B
        eigval, eigvec, res = power_method(B, tol, maxit, x_0)
        
        eigenvalues.append(eigval)
        eigenvectors.append(eigvec)
        residuums.append(res)
        
        # Deflating the matrix B
        B = B - eigval*np.outer(eigvec, np.conjugate(eigvec))
        
    return eigenvalues, eigenvectors, residuums


# Input Parameters for Task b
tol = 1e-5
maxit = 10000
x_0 = None
n = 10
names = ['$B_r$', '$B_g$', '$B_b$']


# Calculating the n largest eigenvalues of B and plotting the residuum
for i, B_i in enumerate(list([B_r, B_g, B_b])):
    
    eigenvalues, eigenvectors, residuums = largest_eigvals(B_r, n, tol, maxit, x_0)
    
    # Logarithmic plot
    plt.figure(figsize = (20,8), dpi = 300)
    plt.yscale('log')  
    plt.xlabel('Iteration Number $p$', fontsize = 14)
    plt.ylabel('$r$ / 1', fontsize = 14)
    plt.title('Residuums against Iterations of the n = %g greatest eigenvalues - ' %n + names[i], fontsize = 18)
    plt.grid(alpha = 0.5)
    for i in range(n):
        plt.plot(list(range(len(residuums[i]))), residuums[i], linewidth = 0.7, 
                 label = '$r_{%g}^{(p)}$' %i)
        plt.legend(fontsize = 14)


#################################    
#################################
#################################

# c) Single Value Decomposition 


def gram_schmidt_modified(u_list):
    """
    
    Parameters
    ----------
    u_list ... list of non-orthonormalized vectors

    Returns
    -------
    v_list ... orthonormalized vectors

    """
    
    v_list = []
    
    for u_k in u_list:
        
        # Copy is necessary because the u_k would change too!
        v_k = np.copy(u_k)
        
        for v_n in v_list:
            
            # Implemeting (8) from excercise sheet
            v_k = v_k - (np.dot(v_k, v_n)/np.dot(v_n, v_n))*v_n
        
        v_k = v_k/np.linalg.norm(v_k)
        v_list.append(v_k)
    
    return v_list
   
        

def svd(B, A, n, tol, maxit, x_0):
    """

    Parameters
    ----------
    B     ... A_conj*A
    A     ... matrix with dimensions (height, length) containing the intensities of a color
    n     ... number of calculated eigenvalues
    tol   ... tolerance for power method
    maxit ... maximum number of iterations for power method
    x_0 : ... starting vector, may be set to None

    Returns
    -------
    A_n   ... decomposited matrix of A

    """
    
    B_eigval, B_eigvec, B_res = largest_eigvals(B, n, tol, maxit, x_0)
    # Making sure that the eigenvalues of B are orthonormal
    B_eigvec_orth = gram_schmidt_modified(B_eigvec)
    # The columns of V are the orthonormal eigenvectors of B
    V = np.column_stack(B_eigvec_orth)
    
    # Calculation of Lambda --> diagonal matrix with roots of eigenvalues of B
    LAMBDA = np.identity(n, dtype = np.cdouble)
    LAMBDA_inv = np.identity(n, dtype = np.cdouble)
    
    for i in range(n):
        LAMBDA[i][i] = np.sqrt(B_eigval[i])
    
    for i in range(n):
        LAMBDA_inv[i][i] = 1/np.sqrt(B_eigval[i])
    
    # U = A * V * Lambda_inverse
    U = np.dot(A, np.dot(V, LAMBDA_inv))
    
    # Calculating the decomposition of A
    A_n = np.dot(U, np.dot(LAMBDA, np.conjugate(np.transpose(V))))
    # Limiting the possible values of A_n to the interval [0, 255]
    # Due to numerical errors there are numbers out of this range
    A_n = np.clip(A_n, 0, 255)
    return A_n


# Input parameters for task c
tol = 1e-5
maxit = 1000000
x_0 = None
n_list = [1, 5, 10, 20, 100]

for n in n_list:
    
    # Calculating the channels for each color
    A_rn = svd(B_r, A_r, n, tol, maxit, x_0)
    A_gn = svd(B_g, A_g, n, tol, maxit, x_0)
    A_bn = svd(B_b, A_b, n, tol, maxit, x_0) 

    # Putting the three channels to an array (height, length, rgb = 3)
    im_compressed = np.zeros((A_rn.shape[0], A_rn.shape[1], 3))

    for i in range(A_rn.shape[0]):
        for j in range(A_rn.shape[1]):
            im_compressed[i][j][0] = A_rn[i][j]
            im_compressed[i][j][1] = A_gn[i][j]
            im_compressed[i][j][2] = A_bn[i][j]
      
    # Type uint8 to convert to image with PIL
    im_compressed = im_compressed.astype(np.uint8)
        
    IMAGE = Image.fromarray(im_compressed)
    IMAGE.save("Codice_Atlantico_n_" + str(n) + ".png")
    

#################################    
#################################
#################################

# d) Single Value Decomposition 

nr_eigvals_B_r = len(np.linalg.eigvals(B_r))
nr_eigvals_B_g = len(np.linalg.eigvals(B_g))
nr_eigvals_B_b = len(np.linalg.eigvals(B_b))

print('Dimensions of the B Matrices: ', B_r.shape)
print('Number of eigenvalues: ', nr_eigvals_B_r)

"""
The Matrices B_r, B_g, B_b have dimensions (1650, 1650). 
Therefore they have each (a maximum of) 1650 eigenvalues.
In c) we added n = 100 and the picture is reasonably sharp with much less 
eigenvalues. 
Storage space original: 5968 KB
Storage space n = 100: 3016 KB
"""

        
    
      
      
      
      
      
      


