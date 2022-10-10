##01a - A simple Pendulum
#Runge-Kutta 4th order methode to solve linear-diff.eq of order 1

# Finds value of y for a given t using specific step size eps
# and initial value y0 at t0.
def runge_Kutta(t0, y0, t, eps):
    # Count number of iterations using step size or
    # step height eps
    n = int( (t - t0) / eps)
    # Iterate for number of iterations
    y = y0
    for i in range(1, n + 1):
        "Apply Runge Kutta Formula for finding next value of y"
        #a_ij, c_j and b_j from Butcher tableau 
        k1 = eps * dydt(y, t)
        k2 = eps * dydt(y + 0.5 * k1, t0 + 0.5 * eps)
        k3 = eps * dydt(y + 0.5 * k2, t0 + 0.5 * eps)
        k4 = eps * dydt(y + k3, t0 + eps)

        #y-parameter updating process
        y = y + 1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4 
        #t-parameter updating process
        t0 = t0 + eps
    return y
 
# initialization 
t0 = 0
y = 1
t = 2
eps = 0.2

#function handling

input_de = input('Write a formula involving t,y : dydt = ')

def dydt(y, t):
    return eval(input_de)

def RK_solver():
    print('The value of y at t for the differential equation dydt =', input_de, 'is:', runge_Kutta(t0, y, t, eps))
    return

RK_solver()