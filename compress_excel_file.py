# This piece of code basically adds the parent directory to PYTHONPATH
import os, sys
parent = os.path.dirname(os.getcwd())
sys.path.append(parent)

# Import numpy, matplotlib and weltgeist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
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

def compress_file(file_name):
    """
    This function return a dictionary of the HII radii from a file
    """
    file = pd.read_excel(r'C:\Users\Hiele\Documents\School\Jaar 5 UvA 2019-20\Scriptie\Thesis\Thesis\measurements/' + file_name + '.xlsx')

    # extract all data from original file
    radii = pd.DataFrame(file, columns= ['Radius edge (m)']).values
    time = pd.DataFrame(file, columns= ['Time (s)']).values
    radii_compressed = []
    time_compressed = []
    previous_radius = 0
    for i in range(len(radii)):
        current_radius = radii[i][0]
        if current_radius != previous_radius:
            radii_compressed.append(radii[i][0])
            time_compressed.append(time[i][0])
            previous_radius = current_radius

    # create excel file
    file = "measurements/" + file_name + "   compressed.xlsx"
    workbook = xlsxwriter.Workbook(file)
    sheet = workbook.add_worksheet()

    # create titels in first row
    sheet.write(0, 0, "Time (s)")
    sheet.write(0, 1, "Radius edge (m)")
    sheet.write(0, 3, "Time (Myr)")
    sheet.write(0, 4, "Radius edge (pc)")

    # widen the columns to make the text more readable
    sheet.set_column(0, 0, 15)
    sheet.set_column(0, 1, 15)
    sheet.set_column(0, 3, 15)
    sheet.set_column(0, 4, 15)

    # write all data in the compressed excel file
    for i in range(len(radii_compressed)):
        sheet.write(i + 1, 0, time_compressed[i])
        sheet.write(i + 1, 1, radii_compressed[i])
        sheet.write(i + 1, 3, round(int(time_compressed[i]) / Myr, 2))
        sheet.write(i + 1, 4, round(radii_compressed[i]/pc, 2))

    workbook.close()

compress_file("n0=1000, rmax=30pc, cool_off, grav_off, T0=10")














    #
