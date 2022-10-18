#Double Pendulum D.E. Function for CP Project

import numpy as np

def Double_Pendulum(D, t, l = 1, g = 9.8067):
    """
    Returns the system of ODE's of a double pendulum with:
        * m = 1
        * l = 1
        * g = 9.8067
    The output is a numpy array
    """

    omega = np.sqrt(g/l)
    theta1, theta2, p1, p2 = D
    
    denominator = (1 + np.sin(theta1 - theta2)**2)
    theta1_der = (p1 - p2*np.cos(theta1 - theta2)) / denominator
    theta2_der = (2*p2 - p1*np.cos(theta1 - theta2)) / denominator
    
    A = (p1*p2*np.sin(theta1 - theta2)) / denominator
    B = ((p1**2 + 2*p2**2 - 2*p1*p2*np.cos(theta1 - theta2))*np.sin(theta1 - theta2)*np.cos(theta1 - theta2)) / (denominator**2) 
    
    p1_der = B - A - 2*omega**2*np.sin(theta1)
    p2_der = A - B - omega**2*np.sin(theta2)
    
    return np.array([theta1_der, theta2_der, p1_der, p2_der])