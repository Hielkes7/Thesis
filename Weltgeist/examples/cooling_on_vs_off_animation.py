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

def find_max(file):
    """
    This function finds the maximum value of the given panda dataframe.
    """
    max_value = pd.DataFrame(file, columns = ['0 iterations']).values[0][0]
    for column_name in file:
        if column_name != 'Radius':
            nH_values = pd.DataFrame(file, columns = [column_name]).values
            for nH_value in nH_values:
                if nH_value[0] > max_value:
                    max_value = nH_value[0]
    return max_value

def animate_xlsx(file_name1, file_name2):

    # opening data files
    file1 = pd.read_excel(r'C:\Users\Hiele\Documents\School\Jaar 5 UvA 2019-20\Scriptie\Weltgeist - edit\examples\data/' + file_name1 + '.xlsx')
    file2 = pd.read_excel(r'C:\Users\Hiele\Documents\School\Jaar 5 UvA 2019-20\Scriptie\Weltgeist - edit\examples\data/' + file_name2 + '.xlsx')

    #TODO: check if file1 and file2 have the same radius step, same niterations, ect

    # subtracting necessary parameters
    amount_of_frames = int(file1.count(axis='columns').values[0])
    amount_of_frames2 = int(file1.count(axis='columns').values[0])
    if amount_of_frames != amount_of_frames2:
        print("Error: files don't have the same dimensions.")
        raise RuntimeError


    # step_iterations = int(file1.columns[2].split(" ")[0])
    # niterations = int(file1.columns[-1].split(" ")[0])
    rmax = pd.DataFrame(file1, columns= ['Radius']).values[-1][0]

    # find the maximum value to adjust plot dimensions to.
    max_value1 = find_max(file1)
    max_value2 = find_max(file2)
    if max_value1 > max_value2:
        max_value = max_value1
    else:
        max_value = max_value2

    # Keep 20% space in the plot dimensions to make it look nicer
    ymax = max_value * 1.2
    # ymax = 2e4 # hardcode ylim

    # start of the animation part
    fig = plt.figure()
    ax = plt.axes(xlim=(0, rmax), ylim=(0, ymax))

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

    lines = []
    line1 = ax.plot([],[], label = file_name1, lw=2, color="blue")[0]
    line2 = ax.plot([],[], label = file_name2, lw=2, color="red")[0]
    lines = [line1, line2]
    # for index in range(2):
    #     line = ax.plot([],[], label = 'test {}'.format(index), lw=2, color=colors[index])[0]
    #     lines.append(line)

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(column_number):
        radii = file1.iloc[ :, 0].values
        nH_file1 = file1.iloc[ :, column_number]
        nH_file2 = file2.iloc[ :, column_number]
        column_number += 1

        xlist = [radii, radii]
        ylist = [nH_file1, nH_file2]

        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum])

        return lines

    column_numbers = np.arange(1, amount_of_frames, 1).tolist()
    FuncAnimation(fig, animate, init_func=init, frames=column_numbers, interval=10, blit=True)
    plt.title("Particle density, nH, per radius")
    plt.xlabel("Radius / pc")
    plt.ylabel("$n_{H}$ / cm$^{-3}$")
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    animate_xlsx(file_name1 = "cooling_on", file_name2 = "cooling_off")
