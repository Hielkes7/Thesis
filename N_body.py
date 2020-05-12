import numpy as np
import random
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

class State():
    """
    This class is the current state of a set up involving a sun and x amount
    of particles.
    """
    def __init__(self, amount_particles, dt, xlim, ylim, particle_size, \
                 mass_density, particle_density):
        self.amount_particles = amount_particles
        self.particles = []
        self.xlim = xlim
        self.ylim = ylim
        self.time = 0
        self.dt = dt
        self.rho = mass_density # kg m^-3
        self.begin_time = time.time()

        # toggle variables, they get toggled on somewhere else
        self.gravity_star_on = False
        self.gravity_particles_on = False
        self.radiation_pressure_on = False
        self.gas_pressure_on = False

        if particle_density == "constant":
            self.constant_particle_density = True
            self.power_law_particle_density = False
        if particle_density == "power_law":
            self.constant_particle_density = False
            self.power_law_particle_density = True

        # create all particles
        # random.seed(0)
        for _ in range(self.amount_particles):
            r_max = xlim[1]
            if self.constant_particle_density:
                r = random.random() * r_max
            if self.power_law_particle_density:
                r = 1/ random.random()**2 * r_max / 100
            theta = random.random() * 2 * np.pi
            x = r * np.cos(theta) # m
            y = r * np.sin(theta) # m
            vx = (random.random()-0.5) * 1e4 # m s^-1
            vy = (random.random()-0.5) * 1e4 # m s^-1
            R = particle_size # m
            m = 4/3 * np.pi * self.rho * R**3
            particle = Particle(x, y, vx, vy, m, R)
            self.particles.append(particle)


    def initiate_star(self, R_star, M_star, QH):
        """
        This function initializes a star in the middle of the 2D plot
        """
        self.QH = QH # amount of ionizing photons
        star = Particle(0, 0, 0, 0, M_star, R_star)
        self.star = star

    def Step(self):
        """
        This function executes one timestep of length dt using Eulers method.
        """
        # reset accelerations of particles from previous iterations
        for particle in self.particles:
            particle.ax = 0
            particle.ay = 0

        # calculate all forces and acceleration
        if self.gravity_star_on:
            self.Gravity_star()
        if self.gravity_particles_on:
            self.Gravity_particles()
        if self.radiation_pressure_on:
            self.Radiation_pressure()
        if self.gas_pressure_on:
            self.Gas_pressure()

        # calculate new positions for each particle
        for particle in self.particles:
            particle.vx += particle.ax * self.dt
            particle.vy += particle.ay * self.dt

            particle.x += particle.vx * self.dt
            particle.y += particle.vy * self.dt

        # update time
        self.time += self.dt

    def Plot(self):
        """
        This function makes a plot of the current state of the set up
        """
        plt.scatter(self.star.x, self.star.y, s=10919 * self.star.R**2, facecolor = "red")
        for particle in self.particles:
            plt.scatter(particle.x, particle.y, s=10919 * particle.R**2, facecolor = "red")
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.show()

    def Radiation_pressure(self):
        """
        This function calculates the acceleration of the particles due to the
        radiation pressure by the star. TODO: The formula used to calculate the
        radiation pressure is incorrect, find and apply the actual formula
        """
        for particle in self.particles:
            dr, dx, dy = self.Distance(particle, self.star)
            # a = 6.25e-50 * self.QH / dr**3
            a = 5e30 / dr**3
            particle.ax += -a * dx / dr
            particle.ay += -a * dy / dr

    def Gas_pressure(self):
        """
        This function calculates the acceleration between particles due to gas
        pressure. TODO: right now the acceleration is equal to the gravity but
        then reversed. Find and apply the actual gas pressure acceleration
        """
        G = 6.67408e-11 # m^3 kg^-1 s^-2

        i = 0
        while i < len(self.particles):
            j = i + 1
            while j < len(self.particles):
                dr, dx, dy = self.Distance(self.particles[i], self.particles[j])
                a1 = G * self.particles[j].m / dr**2
                a2 = G * self.particles[i].m / dr**2
                self.particles[i].ax += -a1 * dx / dr
                self.particles[i].ay += -a1 * dy / dr
                self.particles[j].ax += a2 * dx / dr
                self.particles[j].ay += a2 * dy / dr
                j += 1
            i += 1

    def Gravity_star(self):
        """
        This function calculate the acceleration of all particles due to the
        gravitational pull of the star
        """
        G = 6.67408e-11 # m^3 kg^-1 s^-2

        for particle in self.particles:
            dr, dx, dy = self.Distance(particle, self.star)

            # absorption of particle by star
            if dr < self.star.R + particle.R:
                self.particles.remove(particle)

            # if no absorption, calculate acceleration due to gravity
            else:
                a = G * self.star.m / dr**2
                particle.ax += a * dx / dr
                particle.ay += a * dy / dr

    def Gravity_particles(self):
        """
        This function calculate the acceleration of all particles due to the
        gravitational pull of the other particles
        """
        G = 6.67408e-11 # m^3 kg^-1 s^-2

        i = 0
        while i < len(self.particles):
            j = i + 1
            while j < len(self.particles):
                dr, dx, dy = self.Distance(self.particles[i], self.particles[j])
                if not self.Collision(self.particles[i], self.particles[j], dr):
                    a1 = G * self.particles[j].m / dr**2
                    a2 = G * self.particles[i].m / dr**2
                    self.particles[i].ax += a1 * dx / dr
                    self.particles[i].ay += a1 * dy / dr
                    self.particles[j].ax += -a2 * dx / dr
                    self.particles[j].ay += -a2 * dy / dr
                j += 1
            i += 1

    def Distance(self, particle1, particle2):
        """
        This function returns the distances between the two given particles
        """
        dx = particle2.x - particle1.x
        dy = particle2.y - particle1.y
        dr = ((dx)**2 + (dy)**2)**0.5
        return dr, dx, dy

    def Collision(self, particle1, particle2, dr):
        """
        This function checks if there have been particles which collided, it will
        then create a new particle of combined mass, momentum and new corresponding
        size. "particle1" will be updated, "particle2" will be removed.
        """
        if dr < 0.1 * particle1.R + particle2.R:
            particle1.x = (particle1.x *particle1.m + particle2.x * particle2.m)\
                          / (particle1.m + particle2.m)
            particle1.y = (particle1.y *particle1.m + particle2.y * particle2.m)\
                          / (particle1.m + particle2.m)
            particle1.vx = (particle1.vx *particle1.m + particle2.vx * particle2.m)\
                           / (particle1.m + particle2.m)
            particle1.vy = (particle1.vy *particle1.m + particle2.vy * particle2.m)\
                           / (particle1.m + particle2.m)
            particle1.m = particle1.m + particle2.m
            particle1.R = (3/4 * particle1.m / np.pi / self.rho)**(float(1)/3)
            self.particles.remove(particle2)
            return True
        return False

    def __str__(self):
        return f"{self.amount_particles} particles in a {self.xlim[1]}x \
               {self.ylim[1]} box"


class Particle():
    """
    This class contains information about a particle
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
      return f"A particle of mass {self.m} and radius \
             {self.R} at position ({self.x}, {self.y})"


def animation(animate_live, make_GIF, state, amount_of_frames, niterations):
    """
    This function animates evolution of the set up. There is an option for live
    animation or the making of GIF
    """
    # Some constants
    AU = 1.49597871e11 # meters in an Astronomical Unit
    hour = 3600 # sec in a hour
    day = 86400 # sec in a day
    month = 2.629744e6 # sec in a month
    yr = 3.1556926e7 # sec in a year
    kyr = 3.1556926e10 # sec in a kilo year
    Myr = 3.1556926e13 # sec in a Mega year
    Gyr = 3.1556926e16 # sec in a Giga year

    # simulation settings
    size_box = 5 * AU # diameter of orbit of pluto

    # animation settings
    xlim = -size_box/2, size_box/2
    ylim = -size_box/2, size_box/2
    step_size = int(niterations / amount_of_frames) # iterations done between each frame

    x = []
    y = []

    # add star
    x.append(state.star.x)
    y.append(state.star.y)

    # add all particles
    for particle in state.particles:
        x.append(particle.x)
        y.append(particle.y)

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
        offsets.append([state.star.x, state.star.y])
        sizes.append(1.24e6 * state.star.R**2 * size_box**(-2))
        # 1.24e6 is determined by just trying and it works for having 10
        # inch plot dimensions

        # animate particles
        for particle in state.particles:
            offsets.append([particle.x, particle.y])
            sizes.append(1.24e6 * particle.R**2 * size_box**(-2))
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
    # Some constants
    AU = 1.49597871e11 # meters in an Astronomical Unit
    yr = 3.15576e7 # seconds in a year
    month = yr / 12 # seconds
    Gyr = yr * 1e9 # secondes in a Giga year

    # simulation settings
    time_frame = 0.25 * yr # seconds, about the age of our solar system
    niterations = int(12000)
    size_box = 5 * AU # diameter of orbit of pluto

    # animation settings
    xlim = -size_box/2, size_box/2
    ylim = -size_box/2, size_box/2
    amount_of_frames = int(600)
    dt = time_frame / niterations # s

    # star settings
    R_star = size_box / 100 # radius sun displayed in anmiation, not in scale
    M_star = 1.989e30# mass sun in kg
    QH = 1e45 # photon per second emitted

    # particle settings
    amount_particles = 100
    particle_size = size_box / 1000 # meters,
    # Each particle is still about 1000x larger than earth. However this is the
    # minimum size which will be visible in the animation
    mass_density = 1 # kg m^-3 (rock ish)

    # choose one
    particle_density = "constant"
    # particle_density = "power_law"

    # initializing begin state
    state = State(amount_particles, dt, xlim, ylim, particle_size, mass_density,\
                  particle_density)
    state.initiate_star(R_star, M_star, QH)

    # toggle force parameters
    state.gravity_star_on = True
    # state.gravity_particles_on = True
    # state.radiation_pressure_on = True
    # state.gas_pressure_on = True
    # TODO state.clump_evaportation_on = True
    # TODO state.stellar_wind_on = True

    # Choose either one, they can't both be True
    make_GIF = False
    animate_live = True
    animation(animate_live, make_GIF, state, amount_of_frames, niterations)


if __name__ == "__main__":
    set_up()
