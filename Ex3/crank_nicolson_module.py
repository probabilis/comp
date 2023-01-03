# Crank-Nicolson Module


import numpy as np


def crank_nicolson(Nx, dx, Nt, dt, psi0, V, V0, m):
    
    """
    input parameters:
        
    * Nx   ... number of spatial grid points
    * dx   ... distance between two adjacent grid points
    * Nt   ... maximum number of time steps
    * dt   ... size of time step
    * psi0 ... wave function at t = 0 as the initial condition
    * V    ... a function of the potential
    * V0   ... height of the potential steps 
    * m    ... parameter of the gaussian wave packet
    
    output paramters:
    
    * psi  ... wave function psi(x,t) as 2D-array, rows = time, columns = space 
    """
    
    i = complex(0,1)
    x_values = np.arange(-Nx*dx/3, 2*Nx*dx/3, dx)
    
    # Dimensions of the psi-output, Nt = number of steps -> Nt+1 points in time 
    x_dim = Nx
    t_dim = Nt + 1
    
    # Initialising the psi-array, cdouble to handle complex wave functions
    psi = np.zeros((t_dim, x_dim), dtype = np.cdouble)
    
    # 1) Initial and boundary conditions
    psi[0] = psi0(x_values)
    psi[0][0] = 0
    psi[0][Nx-1] = 0
    
    # 2) Calculating a-array (a0 = aN = 0, not needed but for simpler looping
    #    in same spatial dimension as psi). 
    # Important: N = Nx - 1 (0, 1, ..., N = N + 1 points)
    a = np.zeros(x_dim, dtype = np.cdouble)
    a[1] = 2*(1 + m*dx**2*V(x_values[1], V0) - i*2*m*dx**2/dt)
    
    for k in range(2, Nx - 1):
        term =  2*(1 + m*dx**2*V(x_values[k], V0) - i*2*m*dx**2/dt)
        a[k] = term - 1/a[k-1]
          
    # Initialsing omega and b arrays with same dimensions as psi
    # omega0, omegaN, b0, bN are set to zero but unused
    omega = np.zeros((t_dim, x_dim), dtype = np.cdouble)
    b = np.zeros((t_dim, x_dim), dtype = np.cdouble)
    
    # 3) Time loop 
    for n in range(0, Nt):
        
        # 4) Calculating the omega_nk 
        for k in range(1, Nx - 1):
            omega[n][k] = (- psi[n][k-1] + 2*(i*2*m*dx**2/dt + 1 + 
                           m*dx**2*V(x_values[k], V0))*psi[n][k] - psi[n][k+1])
    
        # 5) Calculating the b_nk
        b[n][1] = omega[n][1]
        
        for k in range(2, Nx - 1):
            b[n][k] = b[n][k-1]/a[k-1] + omega[n][k]
            
        # 6) Calculating the psi_nk with boundary condition = 0
        psi[n+1][0] = 0
        psi[n+1][Nx-1] = 0
        for k in range(Nx - 2, 0, -1):
            psi[n+1][k] = 1/a[k]*(psi[n+1][k+1] - b[n][k])
            
    return psi

