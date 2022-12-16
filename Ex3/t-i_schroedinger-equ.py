#time-independent schrödinger equation

import numpy as np
import matplotlib.pyplot as plt

from runge_kutta import explicit_runge_kutta 


a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])

#############################################
#only for initialization in schroedinger function
eps_en = None
#v_tild(s) is dimenionless transformed value for potential V(x)
#setting zero for inside the potential box (s_min < s < s_max)

def schroedinger(D, s, v_tild = 0):

    psi_1, psi_2 = D 

    psi_1_d = psi_2

    psi_2_d = 2 * (v_tild - eps_en) * psi_1


    return np.array([psi_1_d, psi_2_d])



#inital conditions in phi and phi'
psi_ic = np.array([0, 1])

#intervall in dimensionless parameter s
s_int = [0, 0.5]

#epsilon paramter for runge-kutta
epsilon = 0.001

#nu-threshold for solving psi
nu = 10**(-6)


def solve_schroedinger(psi_ic):
    #set random trial energies
    eps_en_1 = 1 ; eps_en_2 = 2

    global eps_en

    while abs(eps_en_1 - eps_en_2) > nu:

        eps_en = eps_en_1

        psi_a, s_a = explicit_runge_kutta(schroedinger, psi_ic, s_int[0], s_int[1], epsilon, a_rk4, b_rk4, c_rk4)

        eps_en = eps_en_2
        
        psi_b, _ = explicit_runge_kutta(schroedinger, psi_ic, s_int[0], s_int[1], epsilon, a_rk4, b_rk4, c_rk4)

        psi_a = psi_a[0] ; psi_b = psi_b[0]

        if psi_a[-1] * psi_b[-1] > 0:

            eps_en_1 = np.random.randint(0,10000)
            eps_en_2 = np.random.randint(0,10000)

        else:
            eps_en_mid = (eps_en_1 + eps_en_2) / 2

            eps_en = eps_en_mid

            psi_m, _ = explicit_runge_kutta(schroedinger, psi_ic, s_int[0], s_int[1], epsilon, a_rk4, b_rk4, c_rk4)

            psi_m = psi_m[0]

            if psi_a[-1] * psi_m[-1] < 0:
                eps_en_2 = eps_en_mid
            elif psi_b[-1] * psi_m[-1] < 0: 
                eps_en_1 = eps_en_mid
            

    return psi_a, s_a


y = solve_schroedinger(psi_ic)
#print(y)

s_total = np.concatenate((-np.flip(y[1]),y[1]))
y_total = np.concatenate((-y[0],y[0]))

#normalization

I_0 = len(y_total) * np.sum(abs(y_total[:])**2)
#print(I_0)

y_norm = y_total / np.sqrt(2*I_0)

plt.title('time-independent Schrödinger equation for infinite potential-wall')
plt.plot(s_total, y_norm, color = 'gray', label = '$\\psi(x)$')
#plotting potential barriers
plt.vlines(-s_int[1],np.min(y_norm),np.max(y_norm), color = 'black', label = '$\\infty$-potential walls')
plt.vlines(s_int[1],np.min(y_norm),np.max(y_norm), color = 'black')
plt.legend()
plt.xlabel('s / 1')
plt.ylabel('$\\psi(x)$ / 1')
