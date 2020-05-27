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

# Some constants
AU = 1.49597871e11 # meters in an Astronomical Unit
pc = 3.08567758e16 # meters in a parsec
hour = 3600 # sec in a hour
day = 86400 # sec in a day
month = 2.629744e6 # sec in a month
yr = 3.1556926e7 # sec in a year
kyr = 3.1556926e10 # sec in a kilo year
Myr = 3.1556926e13 # sec in a Mega year
Gyr = 3.1556926e16 # sec in a Giga year
mass_sun = 1.989e30 # kg
R_sun = 6.9634e8 # meters
G = 6.67408e-11 # m^3 kg^-1 s^-2, the gravitational constant
u = 1.660539e-27 # kg, atomic mass unit

def animation():

    # setting up all the data
    ncells = 256
    # niterations = 500
    step_iterations = 10
    n0 = 1000.0 # cm^-3
    T0 = 10.0 # K
    rmax = 20.0 * wunits.pc # 20 pc box
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

    nanalytic = np.zeros((ncells))
    hydro = integrator.hydro
    weltgeist.sources.Sources().MakeRadiation(QH)
    # weltgeist.cooling.cooling_on = True
    # weltgeist.gravity.gravity_on = True

    # start of the animation part
    fig = plt.figure()
    ax = plt.axes(xlim=(0, rmax), ylim=(0, ymax))

    colors = ["blue","red"]
    lines = []
    for index in range(2):
        line = ax.plot([],[], lw=2, color=colors[index])[0]
        lines.append(line)

    # creating TICKS for x axis
    amount_of_pc = int(rmax / pc) + 1
    max_amount_ticks = 10
    factor_pc = int(amount_of_pc / max_amount_ticks) + 1
    amount_of_ticks = int(amount_of_pc / factor_pc) + 1
    distance_values = []
    axis_labels = []
    for i in range(amount_of_ticks):
        axis_labels.append(round(i * factor_pc / 100)) # the actual values displayed
        distance_values.append(i * factor_pc * pc) # the location of placement on axis

    ax.set_xticks(distance_values)
    ax.set_xticklabels(axis_labels)

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(frame):
        for _ in range(step_iterations):
            integrator.Step()

        x = hydro.x[:ncells]
        nH = hydro.nH[:ncells]
        nanalytic[:ncells] = n0
        time = integrator.time
        ri = weltgeist.analyticsolutions.SpitzerSolution(QH,n0,time)
        ni = weltgeist.analyticsolutions.SpitzerDensity(QH,n0,time)
        nanalytic[np.where(x <= ri)] = ni
        nH_analytic = nanalytic[:ncells]

        xlist = [x, x]
        ylist = [nH, nH_analytic]

        # Find peak
        nH_peak = nH[0]
        x_peak = x[0]
        i_peak = 0
        for i in range(len(nH)):
            if nH[i] > nH_peak:
                nH_peak = nH[i]
                x_peak = x[i]
                i_peak = i

        # find first point, left of peak, which is lower than n0
        for i in range(i_peak, 0, -1):
            if nH[i] < n0:
                i_edge = i
                x_edge = x[i]
                break

        print("iterations: ", frame * step_iterations)
        print("time: %.2f Myr" %(int(integrator.time) / Myr))
        print("x_edge: ", round(x_edge/pc/100, 1))
        print()

        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum])

        return lines

    FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)
    plt.title("Particle density, nH, per radius")
    plt.xlabel("Radius / pc")
    plt.ylabel("$n_{H}$ / cm$^{-3}$")
    plt.show()



if __name__ == "__main__":
    animation()
