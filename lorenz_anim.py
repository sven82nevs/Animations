import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# from matplotlib.pyplot import style

# style.use('dark_background')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

lines = sum(
    [ax.plot([], [], [], c=(0.133, 0.627, 0.564), lw=0.5, alpha=0.5)], [])

points = sum(
    [ax.plot([], [], [], 'o', c=(0.847, 0.392, 0.274), markerfacecolor='None', 
             markersize=4, markeredgewidth=0.25)], [])

sigma = 10
rho = 28
beta = 8/3
x0 = [0, 1, 20]
t_s, t_m = 0, 150
t = np.linspace(t_s, t_m, 50000)


def lorenz(x_var, t):
    x = x_var[0]
    y = x_var[1]
    z = x_var[2]

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return [dx_dt, dy_dt, dz_dt]


x_solve = np.array([odeint(lorenz, x0, t)])


def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
    return lines, points


def animate(k):
    '''Borrowed some of this function from Jupyterlab's demo as my animation was very slow.'''
    # Controls the number of points plotted per time step.
    s = 13
    k = (s * k) % x_solve.shape[1]

    for line, point, x_i in zip(lines, points, x_solve):
        x, y, z = x_i[:k].T
        line.set_data(x, y)
        line.set_3d_properties(z)
        point.set_data(x[-1:], y[-1:])
        point.set_3d_properties(z[-1:])

    ax.view_init(30, 0.05 * k)
    fig.canvas.draw()
    return lines, points


ax.set_xlim((-15, 15))
ax.set_ylim((-20, 20))
ax.set_zlim((0, 40))
ax.set_axis_off()
ax.grid(False)

anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1)
plt.show()
