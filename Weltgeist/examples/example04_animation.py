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
    ncells = 512
    # niterations = 500
    step_iterations = 10
    n0 = 100.0 # cm^-3
    T0 = 10.0 # K
    T0 = 10.0 # 10 K
    gamma = 5.0/3.0 # monatomic gas (close enough...)

    # dimensions of the bubble and plot
    rmin = 0
    rmax = 10 * wunits.pc # 20 pc box
    ymin = -0.1
    ymax = 1.1 # height of plot/animation

    integrator = weltgeist.integrator.Integrator()
    integrator.Setup(ncells = ncells,
                     rmax = rmax,
                     n0 = n0,
                     T0 = T0,
                     gamma = gamma)

    hydro = integrator.hydro

    # make source
    # Real stars don't form in completely uniform environments
    # The cores that stars form in have roughly(!) "isothermal" profiles
    # This means that density is proportional to 1/r^2
    # Let's modify the initial density field to reflect this
    r0 = 1.0 * wunits.pc
    hydro.nH[0:ncells] = n0 * (hydro.x[0:ncells] / r0)**(-2.0)
    # Get rid of the singularity at r=0
    hydro.nH[0] = hydro.nH[1]
    # We also have to set the temperature correctly again
    hydro.T[0:ncells] = T0
    # Note that this setup is a bit unstable - as long as feedback
    #  responds faster, that's OK...

    # Now let's add a star.
    # First, and this is very important, set the table location
    # This tells the code where the stellar evolution tracks are
    # This will be different for your computer!
    # If you don't have the tables, email me
    # These tables are for Solar metallicity (Z = 0.014)
    # There are also tables for sub-Solar (Z = 0.002)
    # Depends if you want to model our Galaxy or the LMC/SMC
    weltgeist.sources.singlestarLocation = \
        "..\\..\\StellarSources\\Compressed\\singlestar_z0.014"

    # Second, make a star using these tables
    # This is a 30 solar mass star
    # By default it has all the feedback modes turnes on
    # You can turn them off in the function below
    # e.g. star = TableSource(30.0,radiation=False,wind=True)
    star = weltgeist.sources.TableSource(60.0,radiation=False,wind=True)
    weltgeist.sources.Sources().AddSource(star)

    # Turn cooling on
    weltgeist.cooling.cooling_on = True

    # start of the animation part
    fig = plt.figure()
    ax = plt.axes(xlim=(rmin, rmax), ylim=(ymin, ymax))

    colors = ["blue","red", "black"]
    lines = []
    for index in range(3):
        line = ax.plot([],[], lw=2, color=colors[index])[0]
        lines.append(line)

    

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(i):
        for _ in range(step_iterations):
            integrator.Step()

        x = hydro.x[:ncells]
        nH = hydro.nH[:ncells]
        T = hydro.T[0:ncells]
        xhii = hydro.xhii[0:ncells]

        xlist = [x, x, x]
        ylist = [nH, T, xhii]

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
