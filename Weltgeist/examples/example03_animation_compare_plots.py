# This piece of code basically adds the parent directory to PYTHONPATH
import os, sys
parent = os.path.dirname(os.getcwd())
sys.path.append(parent)

# Import numpy, matplotlib and weltgeist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import time
import weltgeist
import weltgeist.units as wunits # make this easier to type
import pandas as pd

def get_max_value(values):
    """
    This function finds the highest value in the given list and returns that value
    """
    max_value = values[0]
    for value in values:
        if value > max_value:
            max_value = value
    return max_value

def animate_xlsx(file_names):

    amount_of_plots = len(file_names)

    # opening data files
    files = []
    for index in range(amount_of_plots):
        file = pd.read_excel(r'C:\Users\Hiele\Documents\School\Jaar 5 UvA 2019-20\Scriptie\Weltgeist - edit\examples\data/' + file_names[index] + '.xlsx')
        files.append(file)

    #TODO: check if file1 and file2 have the same radius step, same niterations, ect

    # step_iterations = int(files[0].columns[2].split(" ")[0])
    # niterations = int(files[0].columns[-1].split(" ")[0])
    rmax = pd.DataFrame(files[0], columns= ['Radius']).values[-1][0]

    # subtracting necessary parameters
    amount_of_frames = int(files[0].count(axis='columns').values[0])

    # load all values in to a big list
    all_data = []
    for file in files:
        for column_name in file:
            if column_name != 'Radius':
                values = pd.DataFrame(file, columns = [column_name]).values
                for value in values:
                    all_data.append(value[0])

    # find biggest value to set ylim to
    max_value = get_max_value(all_data)
    ymax = max_value * 1.2

    # start of the animation part
    fig = plt.figure()
    ax = plt.axes(xlim=(0, rmax), ylim=(0, ymax))

    # make a line for each file and add them to a list
    colors = ["#FD0202", "#0088FF", "#000000", "#841B7A", "#969696", "#994F00", "#1AFF1A", "#E66100", "#E1BE6A"]
    lines = []
    for i, file in enumerate(files):
        line = ax.plot([],[], label = file_names[i], lw=2, color=colors[i])[0]
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(column_number):
        radii = files[0].iloc[ :, 0].values
        xlist, ylist = [], []

        for file in files:
            radii = file.iloc[ :, 0].values
            nH_file = file.iloc[ :, column_number]
            xlist.append(radii)
            ylist.append(nH_file)

        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum])

        return lines

    column_numbers = np.arange(1, amount_of_frames, 1).tolist()
    myAnimation = animation.FuncAnimation(fig, animate, init_func=init, frames=column_numbers,
                              interval=10, blit=True, repeat=False)
    plt.title("Particle density, nH, per radius")
    plt.xlabel("Radius / pc")
    plt.ylabel("$n_{H}$ / cm$^{-3}$")
    plt.legend(loc="upper left")
    plt.show()

    # save animation as a GIF, you have to comment plt.show()
    # myAnimation.save('T10K_vs_T100K_vs_T500K_T2000K.gif', writer='imagemagick', fps=30)

if __name__ == "__main__":
    # animate_xlsx(["T10K", "T100K", "T500K", "T2000K"])
    animate_xlsx(["T10K", "T100K"])
