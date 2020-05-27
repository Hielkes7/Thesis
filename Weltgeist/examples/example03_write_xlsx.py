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


def write_xlsx(cooling_on, gravity_on, file_name):

    # setting up all the data
    ncells = 256
    niterations = 2000
    step_iterations = 5
    n0 = 1000.0 # cm^-3
    rmax = 10.0 * wunits.pc # 20 pc box
    T0 = 2000.0 # K
    gamma = 5.0/3.0 # monatomic gas (close enough...)
    QH = 1e49 # photons per second emitted, roughly a 35 solar mass star


    integrator = weltgeist.integrator.Integrator()
    integrator._time_code = 0.0
    integrator._dt_code = 0.0
    integrator.Setup(ncells = ncells,
                     rmax = rmax,
                     n0 = n0,
                     T0 = T0,
                     gamma = gamma)

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
    sheet.write(0, 0, "Radius")
    for i in range(int(niterations/step_iterations) + 1):
        sheet.write(0, i + 1, str(i*step_iterations) + " iterations")

    # create radius column
    x = hydro.x[:ncells]
    for i in range(ncells):
        sheet.write(i + 1, 0, x[i]/wunits.pc)

    # add nH values for each iteration step
    xlsx_column = 1
    for iteration in range(niterations + 1):
        integrator.Step()

        if iteration % step_iterations == 0:
            print(file_name, ": ", iteration, "/", niterations, " iterations")

            nH = hydro.nH[:ncells]

            # write data to excel file
            for row in range(ncells):
                sheet.write(row + 1, xlsx_column, nH[row])

            xlsx_column += 1

    workbook.close()

if __name__ == "__main__":
    write_xlsx(cooling_on = False,
               gravity_on = False,
               file_name = "T2000K")
