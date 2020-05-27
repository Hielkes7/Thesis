# This piece of code basically adds the parent directory to PYTHONPATH
import os, sys
parent = os.path.dirname(os.getcwd())
sys.path.append(parent)

# Import numpy, matplotlib and weltgeist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import weltgeist
import weltgeist.units as wunits # make this easier to type
from numpy import random 


def animation():

    # setting up all the data 
    ncells = 256
    # niterations = 500
    step_iterations = 10
    n0 = 1000.0 # cm^-3
    T0 = 10.0 # K
    rmax = 10.0 * wunits.pc # 20 pc box
    ymax = 5e3 # height of plot/animation
    T0 = 10.0 # 10 K
    gamma = 5.0/3.0 # monatomic gas (close enough...)
    QH = 1e49 # photons per second emitted, roughly a 35 solar mass star

    integrator = weltgeist.integrator.Integrator()
    integrator.Setup(ncells = ncells, 
                     rmax = rmax, 
                     n0 = n0,
                     T0 = T0,
                     gamma = gamma)
    
    hydro = integrator.hydro
    weltgeist.sources.Sources().MakeRadiation(QH)
    # weltgeist.cooling.cooling_on = True
    # weltgeist.gravity.gravity_on = True

    # start of the animation part
    fig, ax = plt.subplots()
    line, = plt.plot([], [], 'r-')

    def init():
        ax.set_xlim(0, rmax)
        ax.set_ylim(0, ymax)
        return line,

    def animate(i):
        for _ in range(step_iterations):
            integrator.Step()

        x = hydro.x[:ncells]
        nH = hydro.nH[:ncells]
        line.set_data(x, nH)

        return line,

    FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)
    plt.title("Particle density, nH, per radius")
    plt.xlabel("Radius")
    plt.ylabel("nH")
    plt.show()



if __name__ == "__main__":
    animation()