#01 - time-independent schrödinger equation

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from runge_kutta import explicit_runge_kutta

##############################################
# RK4-coefficients

a_rk4 = np.array([[0, 0, 0, 0],
                  [0.5, 0, 0, 0],
                  [0, 0.5, 0, 0],
                  [0, 0, 1, 0]])

b_rk4 = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])

c_rk4 = np.array([0, 0.5, 0.5, 1])

##############################################
##############################################
# only necessary for initialization in schroedinger function
eps_en = None

# v_tild(s) is a dimenionsless transformed value for potential V(x)
# setting zero for inside the potential box (s_min < s < s_max)

#The schroedinger functions (1-4) are systems of ODE for the RungeKutta method for solving the schrödinger equation
#1 for exercise b) and 2/3/4 for exercise d) (below)
def schroedinger1(D, s, v_tild=0):
    psi_1, psi_2 = D

    psi_1_d = psi_2

    psi_2_d = 2 * (v_tild - eps_en) * psi_1

    return np.array([psi_1_d, psi_2_d])


# inital conditions in phi and phi'
psi_symmetric = [1, 0] ; psi_unsymmetric = [0, 1]
psi_ic = np.array([psi_symmetric, psi_unsymmetric])

# intervall in dimensionless parameter s
s_int = [-0.5, +0.5]

# epsilon paramter for runge-kutta
epsilon = 0.001

# nu-threshold for solving psi
nu = 10 ** (- 10)

#############################################
#############################################
# Numerov shooting method for solving stationary schrödinger equation

def solve_schroedinger(psi_ic, schroedinger1, eps_a, eps_b, s_min, s_max, epsilon):
    """
    input parameters:
        psi_ic ... initial conditions in psi (narray with 2 conditions)
        psi(s) ... schrödinger (wave) function in dimensionless domain-parameter s
        eps_a ... trailing starting energy
        eps_b ... trailing ending energy
        s_min ...  minimal domain-value of dimensionless s
        s_max ... maximal domain-value of dimensionless s

    output:
        psi , s , eps_en ... calculated wavefunction psi(s), domain s, calculated eigenenergie due to shooting method
    """

    # set random trial energies
    eps_en_1 = 1; eps_en_2 = 2

    #setting eps_en as global variable in script for overwriting it in the schroedinger function
    global eps_en

    #main loop of trailing energies until convergence is reached
    while abs(eps_en_1 - eps_en_2) > nu:

        eps_en = eps_en_1

        psi_a, s_a = explicit_runge_kutta(schroedinger1, psi_ic, s_min, s_max, epsilon, a_rk4, b_rk4, c_rk4)

        eps_en = eps_en_2

        psi_b, _ = explicit_runge_kutta(schroedinger1, psi_ic, s_min, s_max, epsilon, a_rk4, b_rk4, c_rk4)

        psi_a = psi_a[0]; psi_b = psi_b[0]

        #condition 1:
        if psi_a[-1] * psi_b[-1] > 0:

            eps_en_1 = np.random.randint(eps_a, eps_b)
            eps_en_2 = np.random.randint(eps_a, eps_b)

        #condition 2:
        else:
            eps_en_mid = (eps_en_1 + eps_en_2) / 2

            eps_en = eps_en_mid

            psi_m, _ = explicit_runge_kutta(schroedinger1, psi_ic, s_min, s_max, epsilon, a_rk4, b_rk4, c_rk4)

            psi_m = psi_m[0]

            if psi_a[-1] * psi_m[-1] < 0:
                eps_en_2 = eps_en_mid
            elif psi_b[-1] * psi_m[-1] < 0:
                eps_en_1 = eps_en_mid

    return psi_a, s_a, eps_en_1

#function for seaching roots of calculated psi(s)
def find_roots(y):
    indices = []

    for i in range(len(y) - 2):
        if y[i] > 0 and y[i + 1] < 0:
            indices.append(i)
        if y[i] < 0 and y[i + 1] > 0:
            indices.append(i)

    return indices

#calculating first five wavefunctions through the found roots of wavefunctions
#for optimization symmetric / asymmetric functions (even / odd)are splitted up
def first_five(solve_schroedinger, find_roots, psi_ic_both):
    all_wavefunctions = np.zeros(5, dtype=object)
    all_eigenenergies = np.zeros(5)

    # starting trail energies
    eps_a = 3; eps_b = 20

    symmetric_iter = -1

    #iterating over the needed 5 solutions (n=1 to n=5)
    for k in range(5):
        calc_finished = False
        symmetric = False

        if (k + 1) % 2 != 0:
            # symmetric solution
            psi_ic = psi_ic_both[0]
            s_min = s_int[0] + s_int[1]
            s_max = s_int[1]

            symmetric = True
            symmetric_iter += 1

        else:
            # asymmetric solution
            psi_ic = psi_ic_both[1]
            s_min = s_int[0]
            s_max = s_int[1]

        #endless iteration until right calculation is finished
        while calc_finished != True:

            y, s, eps_en = solve_schroedinger(psi_ic, schroedinger1, eps_a, eps_b, s_min, s_max, epsilon)
            indices = find_roots(y)
            #print('found roots with indices:', indices)

            #right boundary condition is not met until last value of psi (=psi(+L/2) is smaller than epsilon from RK-method
            if y[-1] < epsilon:
                boundary = True

            else:
                boundary = False

            #calculation of psi(s) has met the conditions for symmetric (= even solution)
            if symmetric == True and len(indices) == symmetric_iter and boundary == True:
                indices = np.concatenate((np.flip(indices), indices))
                y = np.concatenate((np.flip(y), y))
                s = np.concatenate((np.flip(-s), s))

                all_wavefunctions[k] = y
                all_eigenenergies[k] = eps_en

                # increasing trailing energies for next wavefunction of order k
                eps_a = 4 * (k + 2) ** 2;
                eps_b = 6 * (k + 2) ** 2

                calc_finished = True

            #calculation of psi(s) has met the conditions for asymmetric (=odd solution)
            if len(indices) == k and boundary == True:

                all_wavefunctions[k] = y
                all_eigenenergies[k] = eps_en

                # increasing trailing energies for next wavefunction of order k
                eps_a = 4 * (k + 2) ** 2;
                eps_b = 6 * (k + 2) ** 2

                calc_finished = True

            else:
                print('Did not calculate right eigen-function of order n =', (k + 1))

    return all_wavefunctions, s, all_eigenenergies


##########################################################
# c) define plotting layout

fig, axs = plt.subplots(5, 1, figsize = (8,12), sharex = True)

#calculating theoretical wavefunctions and finishing plot
def plotting():
    y, s, ee = first_five(solve_schroedinger, find_roots, psi_ic)

    def psi_regular(s, n, L):
        A = np.sqrt(1 / L)
        #condition for symmetric / asymmetric wavefunctions (even / odd)
        if n % 2 == 0:
            return A * np.sin(n * np.pi / L * s)
        else:
            return A * np.cos(n * np.pi / L * s)

    for i in range(len(y)):
        I_0 = 1 / len(y[i]) * np.sum(abs(y[i]) ** 2)
        y_norm = y[i] / np.sqrt(2 * I_0)

        axs[i].plot(s, y_norm, color='gray', label='$\\psi(s)$')
        axs[i].plot(s, psi_regular(s, i + 1, 2 * s_int[1]), color='salmon', label='$\\psi_{th}(s)$')
        axs[i].vlines(s_int[0], -1, +1, color='black', label='$\\infty$-potential walls')
        axs[i].vlines(s_int[1], -1, +1, color='black')
        axs[i].text(0.3, -0.2, '$\\epsilon_{%1d}$ = %3.2f' % ((i + 1), ee[i]))
        axs[i].set_ylim(-1.1, 1.1)
        axs[i].set_ylabel('$\\psi(s)$ / 1')
        axs[i].legend()

    return


# calling functions for exercise c)


plotting()
fig.suptitle('time-independent Schrödinger equation for infinite potential-wall')
fig.tight_layout()
axs[-1].set_xlabel('s / 1')
plt.show()


##########################################################
##########################################################

# d / e (calculating and plotting potentials)

##########################################################
# d) ODE with potential 1:

def schroedinger2(D, s):
    psi_1, psi_2 = D

    psi_1_d = psi_2

    v_0 = 100

    v_tild = v_0 * np.exp(s ** 2 / 0.08)

    psi_2_d = (v_tild - eps_en) * psi_1

    return np.array([psi_1_d, psi_2_d])


##########################################################
# d) ODE with potential 2:

def schroedinger3(D, s):
    psi_1, psi_2 = D

    psi_1_d = psi_2

    k = 1000
    v_0 = 100
    # step-function approx. by logistic function

    v_tild = v_0 * 1 / (1 + np.exp(-2 * k * (s - 0.3))) * 1 / (1 + np.exp(-2 * k * (0.35 - s)))

    psi_2_d = 2 * (v_tild - eps_en) * psi_1

    return np.array([psi_1_d, psi_2_d])


##########################################################
# d) ODE with potential 3:

def schroedinger4(D, s):
    psi_1, psi_2 = D

    psi_1_d = psi_2

    k = 1000
    v_0 = 100
    # step-function approx. by logistic function

    v_tild = - v_0 * 1 / (1 + np.exp(-2 * k * (0.25 - s))) * 1 / (1 + np.exp(-2 * k * (s + 0.25)))

    psi_2_d = 2 * (v_tild - eps_en) * psi_1

    return np.array([psi_1_d, psi_2_d])


##############################################################
# d/e) calculating and plotting wave-functions and eigen-energies

fig, axs = plt.subplots(3, 5, figsize=(20, 12), sharex=True)

psi_ic = np.array([0, 1])

def calc_and_plot(solve_schroedinger, s_min=-0.5, s_max=0.5, epsilon=0.01):
    y_arr = np.zeros(3, dtype=object)

    #trail energies for v_0 = 10 and v_0 = 100 / determined manually (ref. txt sheet at gitlab)
    #v_0 = 10

    #eps_p1 = [[20, 30], [50, 70], [100, 150], [180, 210], [200, 300]]
    #eps_p2 = [[3, 10], [18, 30], [40, 60], [70, 100], [110, 140]]
    #eps_p3 = [[-10, 10], [10, 20], [30, 50], [70, 90], [100, 130]]

    #v_0 = 100

    eps_p1 = [[100, 150], [200, 240], [300, 360], [420, 470], [580, 590]]
    eps_p2 = [[3, 10], [20, 30], [40, 60], [80, 100], [110, 140]]
    eps_p3 = [[-100, -80], [-60, -50], [-10, 0], [30, 50], [60, 100]]

    schr = [schroedinger2, schroedinger3, schroedinger4]
    eps_p = [eps_p1, eps_p2, eps_p3]

    potential = ['$\\tilde{v}_0 exp(s^2/0.08)$', '$\\tilde{v}_0 \\theta(s - 0.3) \\theta(0.35 - s)$',
                 '$\\tilde{v}_0 \\theta(0.25 - s) \\theta(s + 0.25)$']

    s_ev = np.zeros((3, 5))
    s_ev_2 = np.zeros((3, 5))
    colors = ['gray', 'salmon', 'lightskyblue']
    for i in range(len(schr)):
        for j in range(len(eps_p1)):
            boundary = False

            #condition for calculating and plotting right wavefunction
            while boundary == False:

                y, s, ee = solve_schroedinger(psi_ic, schr[i], eps_p[i][j][0], eps_p[i][j][1], s_min, s_max, epsilon)

                if abs(y[-1]) < 0.1:

                    I_0 = 1 / len(y) * np.sum(abs(y) ** 2)
                    y_norm = y / np.sqrt(2 * I_0)

                    y_arr[i] = y_norm

                    axs[i][j].plot(s, y_norm, color=colors[i], label='$\\psi(s)$')

                    axs[i][j].vlines(s_min, np.min(y_norm), np.max(y_norm), color='black', label='boundary conditions')
                    axs[i][j].vlines(s_max, np.min(y_norm), np.max(y_norm), color='black')
                    axs[i][j].text(np.min(s) + 0.1, np.max(y_norm) * 0.5, '$\\epsilon_{%1d}$ = %3.2f' % (j + 1, ee),
                                   bbox=dict(facecolor='white', alpha=0.5))
                    axs[i][j].set_ylabel('$\\psi(s)$ / 1')
                    axs[i][0].legend(loc='upper left')
                    axs[0][j].set_title('n = %1d' % (j + 1))

                    # <s>
                    s_ev[i][j] = (abs(s_min) + s_max) ** 2 * (1 / len(s) * np.sum(s[:] * abs(y[:]) ** 2))

                    s_ev_2[i][j] = (abs(s_min) + s_max) ** 2 * (1 / len(s) * np.sum((s[:])** 2 * abs(y[:]) ** 2))

                    boundary = True
                else:
                    print('Boundary condition not met at Pot.:', i, 'at n =', (j + 1))

        axs[i][0].text(-0.4, 0.1, '$\\tilde{v}_0(s)$ = ' + potential[i], fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.5))

    # f) calculating <s> and <s**2> - <s>
    # calculation of expectation value in <s>

    df_s_ev = pd.DataFrame(np.round(s_ev, 5), columns=['n=1', 'n=2', 'n=3', 'n=4', 'n=5'])

    # calculation of var(s) through <s^2> - <s>^2

    var_s = s_ev_2 - s_ev ** 2

    df_s_var = pd.DataFrame(np.round(var_s, 5), columns=['n=1', 'n=2', 'n=3', 'n=4', 'n=5'])

    return df_s_ev, df_s_var


df_s_ev, df_s_var = calc_and_plot(solve_schroedinger)
df_s_ev.index = np.arange(1, len(df_s_ev) + 1) ; df_s_var.index = np.arange(1, len(df_s_var) + 1)

fig.suptitle('wave functions from n=1 to n=5 for different potentials $\\tilde{v}_0(s)$ with $\\tilde{v}_0 = 100$', fontsize=16)
fig.tight_layout()
plt.show()

print('<s>')
print(df_s_ev)
print('var(s) = <s^2> - <s>^2')
print(df_s_var)

"""
df_s_ev.to_csv(r'df_expectation-value_s.txt', header=None, sep=' ', mode='a')
df_s_var.to_csv(r'df_variance_s.txt', header=None, sep=' ', mode='a')
"""