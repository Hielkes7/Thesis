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
    ncells = 512
    niterations = 10000
    step_iterations = 1
    n0 = 100.0 # cm^-3
    rmax = 15 * wunits.pc # 20 pc box (times 2 because we want to move the edge)
    ymax = 5e2 # height of plot/animation
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
    workbook = xlsxwriter.Workbook('data/HII region expansion.xlsx')
    sheet = workbook.add_worksheet()

    # change width of all cells to make it more readable
    x = hydro.x[:ncells]
    for i in range(len(x) + 1):
        sheet.set_column(0, i, 15)

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

    def animate(step):
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

        # find front edge of expansion HII region
        x_edge = None # create variable in case "i_edge" isn't found yet
        for i in range(len(nH)):
            # start looping bakcwards (right to left)
            if round(nH[len(nH) - i - 1]) != round(n0):
                i_edge = len(nH) - i - 1
                x_edge = x[i_edge]
                break

        # stop saving data when the border of the box is almost reached
        if x_edge:
            if x_edge > 0.95 * rmax:
                print("File closed")
                workbook.close()
                exit()

        # create titels in first row
        sheet.write(step * 4 + 1, 0, "Time (s)")
        sheet.write(step * 4 + 1, 1, integrator.time)

        sheet.write(step * 4 + 1, 3, "Time (Myr)")
        sheet.write(step * 4 + 1, 4, integrator.time / Myr)

        sheet.write(step * 4 + 2, 0, "Radius (m)")
        for i in range(len(x)):
            sheet.write(step * 4 + 2, i + 1, x[i])

        sheet.write(step * 4 + 3, 0, "Density (cm^-3)")
        for i in range(len(nH)):
            sheet.write(step * 4 + 3, i + 1, nH[i])

        if x_edge:
            print(file_name, ": ", step * step_iterations, " iterations")
            print("time: %.2f Myr" %(int(time) / Myr))
            print("x_edge: %.2f pc" %(round(x_edge/pc/100, 2)))
            print()

        xlist = [x, x]
        ylist = [nH, nH_analytic]

        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum])

        for _ in range(step_iterations):
            integrator.Step()

        return lines

    FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)
    # plt.title("Particle density, nH, per radius")
    plt.xlabel("Distance (pc)")
    plt.ylabel("Temperature")
    plt.show()


if __name__ == "__main__":
    write_xlsx(cooling_on = False,
               gravity_on = False,
               file_name = "HII region expansion")
