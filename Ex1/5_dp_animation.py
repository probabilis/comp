
# 5 - Animation
#reference:
#https://matplotlib.org/stable/gallery/animation/double_pendulum.html

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
#importing the global General-Explicit RK algorithm from the project folder
from runge_kutta import explicit_runge_kutta 
#importing the global Double-Pendulum differential equations 
from double_pendulum_de import Double_Pendulum


#RK4-parameters of the Butcher tableau

a_rk4 = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0.5,0,0],
                  [0,0,1,0]])

b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])

c_rk4 = np.array([0, 0.5, 0.5, 1])


#############################################
#############################################
#############################################

#initial conditions

#input:
############################
theta1_der_0 = 1
theta2_der_0 = 0
p1_der_0 = 0
p2_der_0 = 3
############################
############################
y0 = np.array([theta1_der_0, theta2_der_0, p1_der_0, p2_der_0])
############################

#initializing parameters
t0 = 0
tmax = 10
epsilon = 0.1

L1 = 1 ; L2 = 1
L = L1 + L2
M1 = 1 ; M2 = 1
history_len = 500

def ini_para_for_plot(y0):
    y, t = explicit_runge_kutta(Double_Pendulum, y0, t0, tmax, epsilon, a_rk4, b_rk4, c_rk4)
    theta1, theta2, p1, p2 = y

    x1 = L1*np.sin(theta1)
    y1 = -L1*np.cos(theta1)

    x2 = L2*np.sin(theta2) + x1
    y2 = -L2*np.cos(theta2) + y1
    return x1, y1, x2, y2, y, t


x1, y1, x2, y2, y, t = ini_para_for_plot(y0)

#plotting


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()


    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*epsilon))

    return line, trace, time_text


ani = FuncAnimation(fig, animate, len(t), interval=epsilon*1000, blit=True)

ax.set_title('Double Pendulum Simulation with I.C.: $\\dot{\\theta}_1$ = %1.f, $\\dot{\\theta}_2$ = %1.f, $\\dot{p}_1$ = %1.f, $\\dot{p}_2$ = %.f' %(theta1_der_0, theta2_der_0, p1_der_0, p2_der_0), fontsize = 10)


ani.save("dp_animation.gif", dpi=100, writer=PillowWriter(fps=100))
plt.show()



