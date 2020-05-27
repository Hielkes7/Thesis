"""
Example 2 - Leonid Sedov
Run a simple test problem and compare it to an analytic solution

@author: samgeen
"""

# This piece of code basically adds the parent directory to PYTHONPATH
import os, sys
parent = os.path.dirname(os.getcwd())
sys.path.append(parent)

# Import numpy, matplotlib and weltgeist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import weltgeist
import weltgeist.units as wunits # make this easier to type


def animation():
    
    # setting up all the data 
    ncells = 256
    niterations = 5000
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
    
    nanalytic = np.zeros((ncells))
    hydro = integrator.hydro
    weltgeist.sources.Sources().MakeRadiation(QH)
    # weltgeist.cooling.cooling_on = True
    # weltgeist.gravity.gravity_on = True


    for iteration in range(niterations):
        integrator.Step()

        if (iteration % step_iterations) == 0:

            # analytic line
            x = hydro.x[:ncells]
            nanalytic[:ncells] = n0
            time = integrator.time
            ri = weltgeist.analyticsolutions.SpitzerSolution(QH,n0,time)
            ni = weltgeist.analyticsolutions.SpitzerDensity(QH,n0,time)
            nanalytic[np.where(x <= ri)] = ni
            
            # simulated line
            nH = hydro.nH[:ncells]

            plt.clf()
            plt.cla()
            # plt.plot(radii, temp, color="r")
            plt.plot(x, nH, color="b")
            plt.plot(x, nanalytic, color="r")
            plt.xlim(0,rmax)
            plt.ylim(0, ymax)
            plt.title("Temperature per radius after %d iterations" %(iteration))
            plt.xlabel("Radius")
            plt.ylabel("Temperature")
            plt.draw()
            plt.pause(0.01)


    
if __name__=="__main__":
    animation()
    
