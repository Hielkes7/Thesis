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
import xlsxwriter
import pandas as pd

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


def write_xlsx(cooling_on, gravity_on, file_name):

    # setting up all the data
    ncells = 256
    niterations = 10000
    step_iterations = 10
    n0 = 1000.0 # cm^-3
    rmax = 10.0 * wunits.pc * 2 # 20 pc box (times 2 because we want to move the edge)
    ymax = 5e3 # height of plot/animation
    T0 = 10.0 # K
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
    weltgeist.cooling.cooling_on = cooling_on
    weltgeist.gravity.gravity_on = gravity_on

    # create excel file
    file = "data/" + file_name + ".xlsx"
    workbook = xlsxwriter.Workbook(file)
    sheet = workbook.add_worksheet()

    # widen all columns to make the text clearer
    for i in range(int(niterations/step_iterations) + 1):
        sheet.set_column(0, i, 15)

    # create titels in first row
    sheet.write(0, 0, "Iterations")
    sheet.write(0, 1, "Time (s)")
    sheet.write(0, 2, "Time (Myr)")
    sheet.write(0, 3, "Radius edge (m)")
    sheet.write(0, 4, "Radius edge (pc)")


    # start of the animation part
    fig = plt.figure()
    ax = plt.axes(xlim=(0, rmax), ylim=(0, ymax))

    colors = ["blue","red"]
    lines = []
    for index in range(2):
        line = ax.plot([],[], lw=2, color=colors[index])[0]
        lines.append(line)

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

        # write data to excel file
        if x_edge < 0.75 * rmax:
            sheet.write(frame + 1, 0, frame * step_iterations)
            sheet.write(frame + 1, 1, time)
            sheet.write(frame + 1, 2, round(int(time) / Myr, 1))
            sheet.write(frame + 1, 3, x_edge/100)
            sheet.write(frame + 1, 4, round(x_edge/100/pc, 1))

            print(file_name, ": ", frame * step_iterations, " iterations")
            print("time: %.2f Myr" %(int(time) / Myr))
            print("x_edge: %.2f pc" %(round(x_edge/pc/100, 2)))
            print()

        if x_edge > 0.75 * rmax:
            workbook.close()

        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum])

        return lines

    FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)
    # plt.title("Particle density, nH, per radius")
    plt.xlabel("particle density ($cm_{-3}$)")
    plt.ylabel("Temperature")
    plt.show()

    workbook.close()

if __name__ == "__main__":
    write_xlsx(cooling_on = False,
               gravity_on = False,
               file_name = "n0=1000, rmax=10pc, cool_off, grav_off, T0=10")
