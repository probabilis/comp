import numpy as np

from runge_kutta import explicit_runge_kutta

import matplotlib.pyplot as plt

a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])


def linear(x, a, b):
    return np.array([a * x + b])

y , x = explicit_runge_kutta(linear,0,0,100,0.01,a_rk4,b_rk4,c_rk4)


plt.plot(x,y)

