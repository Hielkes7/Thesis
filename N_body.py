import numpy as np
import random
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

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


class State():
    """
    This class is the current state of a set up involving a sun and x amount
    of clumps.
    """
    def __init__(self, amount_clumps, dt, xlim, ylim, clump_size, \
                 mass_density, clump_density):
        self.amount_clumps = amount_clumps
        self.clumps = []
        self.xlim = xlim
        self.ylim = ylim
        self.time = 0
        self.dt = dt
        self.rho = mass_density # kg m^-3
        self.begin_time = time.time()
        self.star = None

        # toggle variables, they get toggled on somewhere else
        self.gravity_star_on = False
        self.gravity_clumps_on = False
        self.radiation_pressure_on = False
        self.gas_pressure_on = False

        if clump_density == "constant":
            self.constant_clump_density = True
            self.power_law_clump_density = False
        if clump_density == "power_law":
            self.constant_clump_density = False
            self.power_law_clump_density = True

        # create all clumps
        # random.seed(0)
        for _ in range(self.amount_clumps):
            r_max = xlim[1]
            if self.constant_clump_density:
                r = random.random() * r_max
            if self.power_law_clump_density:
                r = 1/ random.random()**2 * r_max / 100
            theta = random.random() * 2 * np.pi
            x = r * np.cos(theta) # m
            y = r * np.sin(theta) # m
            vx = (random.random()-0.5) * 1e4 # m s^-1
            vy = (random.random()-0.5) * 1e4 # m s^-1
            R = clump_size # m
            m = 4/3 * np.pi * self.rho * R**3
            clump = Object(x, y, vx, vy, m, R)
            self.clumps.append(clump)


    def initiate_star(self, R_star, M_star, QH):
        """
        This function initializes a star in the middle of the 2D plot
        """
        self.QH = QH # amount of ionizing photons
        star = Object(0, 0, 0, 0, M_star, R_star)
        self.star = star

    def Step(self):
        """
        This function executes one timestep of length dt using Eulers method.
        """
        # reset accelerations of clumps from previous iterations
        for clump in self.clumps:
            clump.ax = 0
            clump.ay = 0

        # calculate all forces and acceleration
        if self.gravity_star_on:
            self.Gravity_star()
        if self.gravity_clumps_on:
            self.Gravity_clumps()
        if self.radiation_pressure_on:
            self.Radiation_pressure()
        if self.gas_pressure_on:
            self.Gas_pressure()

        # calculate new positions for each clump
        for clump in self.clumps:
            clump.vx += clump.ax * self.dt
            clump.vy += clump.ay * self.dt

            clump.x += clump.vx * self.dt
            clump.y += clump.vy * self.dt

        # update time
        self.time += self.dt

    def Plot(self):
        """
        This function makes a plot of the current state of the set up
        """
        plt.scatter(self.star.x, self.star.y, s=10919 * self.star.R**2, facecolor = "red")
        for clump in self.clumps:
            plt.scatter(clump.x, clump.y, s=10919 * clump.R**2, facecolor = "red")
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.show()

    def Radiation_pressure(self):
        """
        This function calculates the acceleration of the clumps due to the
        radiation pressure by the star. TODO: The formula used to calculate the
        radiation pressure is incorrect, find and apply the actual formula
        """
        for clump in self.clumps:
            dr, dx, dy = self.Distance(clump, self.star)
            # a = 6.25e-50 * self.QH / dr**3
            a = 5e30 / dr**3
            clump.ax += -a * dx / dr
            clump.ay += -a * dy / dr

    def Gas_pressure(self):
        """
        This function calculates the acceleration between clumps due to gas
        pressure. TODO: right now the acceleration is equal to the gravity but
        then reversed. Find and apply the actual gas pressure acceleration
        """
        G = 6.67408e-11 # m^3 kg^-1 s^-2

        i = 0
        while i < len(self.clumps):
            j = i + 1
            while j < len(self.clumps):
                dr, dx, dy = self.Distance(self.clumps[i], self.clumps[j])
                a1 = G * self.clumps[j].m / dr**2
                a2 = G * self.clumps[i].m / dr**2
                self.clumps[i].ax += -a1 * dx / dr
                self.clumps[i].ay += -a1 * dy / dr
                self.clumps[j].ax += a2 * dx / dr
                self.clumps[j].ay += a2 * dy / dr
                j += 1
            i += 1

    def Gravity_star(self):
        """
        This function calculate the acceleration of all clumps due to the
        gravitational pull of the star
        """
        G = 6.67408e-11 # m^3 kg^-1 s^-2

        if not self.star:
            raise Exception("Error, can't calculate gravity of star. There is no star")

        for clump in self.clumps:
            dr, dx, dy = self.Distance(clump, self.star)

            # absorption of clump by star
            if dr < self.star.R + clump.R:
                self.clumps.remove(clump)

            # if no absorption, calculate acceleration due to gravity
            else:
                a = G * self.star.m / dr**2
                clump.ax += a * dx / dr
                clump.ay += a * dy / dr

    def Gravity_clumps(self):
        """
        This function calculate the acceleration of all clumps due to the
        gravitational pull of the other clumps
        """
        G = 6.67408e-11 # m^3 kg^-1 s^-2

        i = 0
        while i < len(self.clumps):
            j = i + 1
            while j < len(self.clumps):
                dr, dx, dy = self.Distance(self.clumps[i], self.clumps[j])
                if not self.Collision(self.clumps[i], self.clumps[j], dr):
                    a1 = G * self.clumps[j].m / dr**2
                    a2 = G * self.clumps[i].m / dr**2
                    self.clumps[i].ax += a1 * dx / dr
                    self.clumps[i].ay += a1 * dy / dr
                    self.clumps[j].ax += -a2 * dx / dr
                    self.clumps[j].ay += -a2 * dy / dr
                j += 1
            i += 1

    def Distance(self, clump1, clump2):
        """
        This function returns the distances between the two given clumps
        """
        dx = clump2.x - clump1.x
        dy = clump2.y - clump1.y
        dr = ((dx)**2 + (dy)**2)**0.5
        return dr, dx, dy

    def Collision(self, clump1, clump2, dr):
        """
        This function checks if there have been clumps which collided, it will
        then create a new clump of combined mass, momentum and new corresponding
        size. "clump1" will be updated, "clump2" will be removed.
        """
        if dr < 0.1 * clump1.R + clump2.R:
            clump1.x = (clump1.x *clump1.m + clump2.x * clump2.m)\
                          / (clump1.m + clump2.m)
            clump1.y = (clump1.y *clump1.m + clump2.y * clump2.m)\
                          / (clump1.m + clump2.m)
            clump1.vx = (clump1.vx *clump1.m + clump2.vx * clump2.m)\
                           / (clump1.m + clump2.m)
            clump1.vy = (clump1.vy *clump1.m + clump2.vy * clump2.m)\
                           / (clump1.m + clump2.m)
            clump1.m = clump1.m + clump2.m
            clump1.R = (3/4 * clump1.m / np.pi / self.rho)**(float(1)/3)
            self.clumps.remove(clump2)
            return True
        return False

    def Radius_clump(Mass):
        """
        This function calculates the radius of clumps depending on their mass.
        I derived this formula of excisting data of mass/radius ratios. See
        the file "radius_mass_ratio_clumps.pdf" on my github:
        https://github.com/Hielkes7/Thesis
        """
        # I derived this formula of excisting data of mass/radius ratios.
        Radius = (1/55.550)* Mass**(0.426)
        return Radius

    def Mass_clump(Radius):
        """
        This function calculates the mass of clumps depending on their radius.
        I derived this formula of excisting data of mass/radius ratios. See
        the file "radius_mass_ratio_clumps.pdf" on my github:
        https://github.com/Hielkes7/Thesis
        """
        # I derived this formula of excisting data of mass/radius ratios.
        Mass = 12590 * Radius**(2.35)
        return Mass

    def __str__(self):
        return f"{self.amount_clumps} clumps in a {self.xlim[1]}x \
               {self.ylim[1]} box"


class Object():
    """
    This class contains information about a clump
    """
    def __init__(self, x, y, vx, vy, m, R):
      self.x = x    # position x coordinate
      self.y = y    # position y coordinate
      self.vx = vx      # velocity in x direction
      self.vy = vy      # velocity in y direction
      self.ax = 0       # acceleration in x direction
      self.ay = 0       # acceleration in y direction
      self.m = m
      self.R = R

    def __str__(self):
      return f"A clump of mass {self.m} and radius \
             {self.R} at position ({self.x}, {self.y})"


def animation(animate_live, make_GIF, state, amount_of_frames, niterations, size_box):
    """
    This function animates evolution of the set up. There is an option for live
    animation or the making of GIF
    """
    # animation settings
    xlim = -size_box/2, size_box/2
    ylim = -size_box/2, size_box/2
    step_size = int(niterations / amount_of_frames) # iterations done between each frame

    x = []
    y = []

    # add star, if there is one
    if state.star:
        x.append(state.star.x)
        y.append(state.star.y)

    # add all clumps
    for clump in state.clumps:
        x.append(clump.x)
        y.append(clump.y)

    fig = plt.figure()
    plt.grid()
    fig.set_size_inches(10, 10) # 10 inches wide and long
    scat = plt.scatter(x, y, facecolor = "red")
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    title = ax.text(0.5, 1.02, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha="center")

    def update(frame):
        # if frame != 0:
        #     # information feedback to estimate duration of animation
        #     print(frame, "/", amount_of_frames, "  Time:", (time.time() - \
        #           state.begin_time)*amount_of_frames/frame, "sec")


        offsets = []
        sizes = []
        title.set_text(u"{} / {} iterations - {} months".format(frame*step_size,\
                       niterations, round(state.time / month, 1)))

        # animate star
        if state.star:
            offsets.append([state.star.x, state.star.y])
            sizes.append(1.24e6 * state.star.R**2 * size_box**(-2))
            # 1.24e6 is determined by just trying and it works for having 10
            # inch plot dimensions

        # animate clumps
        for clump in state.clumps:
            offsets.append([clump.x, clump.y])
            sizes.append(1.24e6 * clump.R**2 * size_box**(-2))
        scat.set_offsets(offsets)
        scat.set_sizes(sizes)
        for _ in range(step_size):
            state.Step()
        return scat,title,

    myAnimation = FuncAnimation(fig, update, frames = amount_of_frames, \
                                interval = 10, repeat=False)
    if animate_live:
        plt.show()
    if make_GIF:
        myAnimation.save('RENAME_THIS_GIF.gif', writer='imagemagick', fps=30)


def set_up():
    """
    This function contains all the input values. The user adjusts them.
    """
    # simulation settings
    time_frame =  10 * yr # seconds, about the age of our solar system
    niterations = int(12000)
    size_box = 8 * pc # diameter of orbit of pluto

    # animation settings
    xlim = -size_box/2, size_box/2
    ylim = -size_box/2, size_box/2
    amount_of_frames = int(600)
    dt = time_frame / niterations # s

    # star settings
    R_star = size_box / 100 # radius sun displayed in anmiation, not in scale
    M_star = 1.989e30# mass sun in kg
    QH = 1e45 # photon per second emitted

    # clump settings
    amount_clumps = 20
    clump_size = size_box / 1000 # meters,
    # Each clump is still about 1000x larger than earth. However this is the
    # minimum size which will be visible in the animation
    mass_density = 1 # kg m^-3 (rock ish)

    # choose one
    clump_density = "constant"
    # clump_density = "power_law"

    # initializing begin state
    state = State(amount_clumps, dt, xlim, ylim, clump_size, \
                  mass_density, clump_density)
    # state.initiate_star(R_star, M_star, QH)

    # toggle force parameters
    # state.gravity_star_on = True
    state.gravity_clumps_on = True
    # state.radiation_pressure_on = True
    # state.gas_pressure_on = True
    # TODO state.clump_evaportation_on = True
    # TODO state.stellar_wind_on = True

    # Choose either one, they can't both be True
    make_GIF = False
    animate_live = True
    animation(animate_live, make_GIF, state, amount_of_frames, niterations, \
              size_box)


if __name__ == "__main__":
    set_up()
