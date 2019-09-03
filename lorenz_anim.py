from matplotlib.pyplot import style
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
import numpy as np

style.use('dark_background')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

lines = sum(
    [ax.plot([], [], [], color=(0.133, 0.627, 0.564), lw=0.25, alpha=0.5)], [])
points = sum(
    [ax.plot([], [], [], 'o', color=(0.847, 0.392, 0.274),
             markerfacecolor='None', markersize=4, markeredgewidth=0.25)], [])

sigma = 10
rho = 28
beta = 8/3
x0 = [0, 1, 15]
t_s, t_m = 0, 100
t = np.linspace(t_s, t_m, 25000)


def lorenz(x, t):
    x1 = x[0]
    y = x[1]
    z = x[2]

    dx_dt = sigma * (y - x1)
    dy_dt = x1 * (rho - z) - y
    dz_dt = x1 * y - beta * z

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

anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=25000, interval=1)
plt.show()
