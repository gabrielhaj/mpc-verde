import numpy as np

import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as pp
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from numpy import sin, cos


def simulate(solution,dt):
    L = 0.5
    ths = solution[2, :]
    xs = solution[0, :]

    pxs = L * sin(ths) + xs
    pys = L * cos(ths)

    fig = pp.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.5, 1.5), ylim=(-0.5, 2))
    ax.set_aspect('equal')
    ax.grid()

    patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    cart_width = 0.3
    cart_height = 0.2

    def init():
        line.set_data([], [])
        time_text.set_text('')
        patch.set_xy((-cart_width/2, -cart_height/2))
        patch.set_width(cart_width)
        patch.set_height(cart_height)
        return line, time_text, patch


    def animate(i):
        thisx = [xs[i], pxs[i]]
        thisy = [0, pys[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        patch.set_x(xs[i] - cart_width/2)
        return line, time_text, patch

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(solution)),
                                interval=25, blit=True, init_func=init)

    pp.show()

    # Set up formatting for the movie files
    print("Writing video...")
    Writer = animation.writers['imagemagick']
    writer = Writer(fps=25, metadata=dict(artist='Sergey Royz'), bitrate=1800)
    ani.save('controlled-cart.gif', writer=writer)
    return    