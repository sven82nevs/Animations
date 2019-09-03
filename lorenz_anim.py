from matplotlib.pyplot import style
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
import numpy as np

style.use('dark_background')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# line = ax.plot([], [], [])

sigma = 10
rho = 28
beta = 8/3
x0 = [0, 1, 15]
t_end, n = 100, 25000
t = np.linspace(0, t_end, n)


def lorenz(x, t):
    x1 = x[0]
    y = x[1]
    z = x[2]

    dx_dt = sigma * (y - x1)
    dy_dt = x1 * (rho - z) - y
    dz_dt = x1 * y - beta * z

    return [dx_dt, dy_dt, dz_dt]


x_solve = np.array([odeint(lorenz, x0, t)])

lines = sum(
    [ax.plot([], [], [], color=(0.133, 0.627, 0.564), lw=0.25, alpha=0.5)], [])
pts = sum([ax.plot([], [], [], 'o', color=(0.847, 0.392, 0.274), markersize=4,
                   markerfacecolor='None', markeredgewidth=0.25)], [])


def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines, pts


def animate(k):
    #  Controls the number of points plotted per time step.
    k = (7 * k) % x_solve.shape[1]

    for line, pt, x_i in zip(lines, pts, x_solve):
        x, y, z = x_i[:k].T
        line.set_data(x, y)
        line.set_3d_properties(z)
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * k)
    fig.canvas.draw()
    return lines + pts


ax.set_xlim((-15, 15))
ax.set_ylim((-20, 20))
ax.set_zlim((0, 40))

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('k')
ax.yaxis.pane.set_edgecolor('k')
ax.zaxis.pane.set_edgecolor('k')
ax.set_axis_off()
ax.grid(False)

anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=5000, interval=1)
plt.show()
