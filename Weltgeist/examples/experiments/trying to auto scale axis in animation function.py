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
    T0 = 3000.0 # 10 K
    gamma = 5.0/3.0 # monatomic gas (close enough...)
    QH = 1e49 # photons per second emitted, roughly a 35 solar mass star

    integrator = weltgeist.integrator.Integrator()
    integrator.Setup(ncells = ncells, 
                     rmax = rmax, 
                     n0 = n0,
                     T0 = T0,
                     gamma = gamma)
    
    nanalytic = np.zeros((ncells))
    hydro = integrator.hydro
    weltgeist.sources.Sources().MakeRadiation(QH)
    weltgeist.cooling.cooling_on = True
    # weltgeist.gravity.gravity_on = True

    # start of the animation part
    fig = plt.figure()
    ax = plt.axes(xlim=(0, rmax), ylim=(0, ymax))

    plotcols = ["blue","red"]
    lines = []
    for index in range(2):
        lobj = ax.plot([],[], lw=2, color=plotcols[index])[0]
        lines.append(lobj)

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(i):
        for _ in range(step_iterations):
            integrator.Step()

        highest_value = np.amax(hydro.nH[:ncells])
        ylim = ax.get_ylim()
        if highest_value > ylim[1]:
            xlim = ax.get_xlim()
            ax.cla()
            ax.set_xlim(xlim)
            ax.set_ylim(0, highest_value * 1.5)
            ax.set_ylabel("test")

        x = hydro.x[:ncells]
        nanalytic[:ncells] = n0
        time = integrator.time
        ri = weltgeist.analyticsolutions.SpitzerSolution(QH,n0,time)
        ni = weltgeist.analyticsolutions.SpitzerDensity(QH,n0,time)
        nanalytic[np.where(x <= ri)] = ni

        xlist = [x, x]
        ylist = [hydro.nH[:ncells], nanalytic[:ncells]]
  
        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum])

        return lines

    FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)
    plt.title("Particle density, nH, per radius")
    plt.xlabel("Radius")
    plt.ylabel("Temperature")
    plt.show()




if __name__ == "__main__":
    animation()