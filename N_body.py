import numpy as np
import random
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

class State():
    def __init__(self, amount_particles, dt, xlim, ylim, particle_size, mass_density):
        self.amount_particles = amount_particles
        self.particles = []
        self.xlim = xlim
        self.ylim = ylim
        self.time = 0
        self.dt = dt
        self.rho = mass_density # kg m^-3
        self.begin_time = time.time()

        self.gravity_on = False

        # create all particles
        # random.seed(0)
        for _ in range(self.amount_particles):
            r_max = (xlim[1]**2 + ylim[1]**2)**0.5
            r = random.random() * r_max
            theta = random.random() * 2 * np.pi
            x = r * np.cos(theta) # m
            y = r * np.sin(theta) # m
            vx = 0 # m s^-1
            vy = 0 # m s^-1
            R = particle_size # m
            m = 4/3 * np.pi * self.rho * R**3
            particle = Particle(x, y, vx, vy, m, R)
            self.particles.append(particle)

    def initiate_star(self, R_star, M_star, QH):
        """
        This function initializes a star in the middle of the 2D plot
        """
        star = Particle(0, 0, 0, 0, M_star, R_star)
        self.particles.append(star)

    def Plot(self):
        for particle in self.particles:
            plt.scatter(particle.x, particle.y, s=10919 * particle.R**2, facecolor = "red")
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.show()

    def Step(self):
        # reset accelerations of particles from previous iterations
        for particle in self.particles:
            particle.ax = 0
            particle.ay = 0

        # calculate all forces and acceleration
        if self.gravity_on:
            self.Gravity()

        # calculate new positions for each particle
        for particle in self.particles:
            particle.vx += particle.ax * self.dt
            particle.vy += particle.ay * self.dt

            particle.x += particle.vx * self.dt
            particle.y += particle.vy * self.dt

        # update time
        self.time += self.dt

    def Gravity(self):
        """
        This function updates the acceleration of each particle due to gravity
        """
        G = 6.67408e-11 # m^3 kg^-1 s^-2

        # calculate the acceleration of all particles due to gravity
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
        size. And remove the old 2 particles
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
        return f"{self.amount_particles} particles in a {self.xlim[1]}x{self.ylim[1]} box"

class Particle():
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

def test():
    amount_particles = 10
    size_box = 10
    dt = 1
    xlim = 0, size_box
    ylim = 0, size_box

    # initializing begin state
    state = State(amount_particles, dt, xlim, ylim)
    state.Plot()
    state.Step()
    state.Plot()

def animation(live, make_GIF):
    # niterations = int(3e1)
    niterations = int(1.2e3)
    amount_particles = 50
    size_box = 100
    dt = 20
    xlim = -size_box/2, size_box/2
    ylim = -size_box/2, size_box/2
    step_size = 20 # amount of iterations done for each next frame
    particle_size = 0.5 # m
    mass_density = 2400 # kg m^-3 (rock ish)

    # initializing begin state
    state = State(amount_particles, dt, xlim, ylim, particle_size, mass_density)

    # turn gravity on
    state.gravity_on = True
    state.star = True

    R_star = 2 # m
    rho_star = 2400 # kg m^-3 (mass density star)
    M_star = 4/3 * R_star**3 * np.pi * rho_star # kg
    QH = 1e45 # photon per second emitted
    # state.initiate_star(R_star, M_star, QH)

    x = []
    y = []
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
        offsets = []
        sizes = []
        title.set_text(u"{} / {} iterations - {} sec".format(frame, niterations, state.time))
        if len(state.particles) == 1:
            print("Tijd: ", state.begin_time - time.time())
        print(frame, " / ", niterations, ",   ", "number of particles: ", len(state.particles))
        for particle in state.particles:
            offsets.append([particle.x, particle.y])
            sizes.append(1.24e6 * particle.R**2 * size_box**(-2))
            # 1.24e6 is determined by just trying and it works for having 10 inch plot dimensions
        scat.set_offsets(offsets)
        scat.set_sizes(sizes)
        for _ in range(step_size):
            state.Step()
        return scat,title,

    myAnimation = FuncAnimation(fig, update, frames = niterations, interval = 10,
                                repeat=False)
    if live:
        plt.show()
    if make_GIF:
        myAnimation.save('animations4.gif', writer='imagemagick', fps=30)

def pauze():
    niterations = int(1e3)
    amount_particles = 5
    size_box = 10
    dt = 20
    xlim = 0, size_box
    ylim = 0, size_box
    step_size = 20 # amount of iterations done for each next frame
    particle_size = 0.025 # m
    mass_density = 2400 # kg m^-3 (rock ish)

    # initializing begin state
    state = State(amount_particles, dt, xlim, ylim, particle_size, mass_density)

    # turn gravity on
    state.gravity_on = True

    x = []
    y = []
    for particle in state.particles:
        x.append(particle.x)
        y.append(particle.y)

    fig = plt.figure()
    fig.set_size_inches(10, 10) # 10 inches wide and long
    scat = plt.scatter(x, y, facecolor = "red")
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # scat = plt.scatter(particle.x, particle.y, s=10919 * particle.R**2, facecolor = "red")

    def update(frame):
        offsets = []
        sizes = []
        print(frame, " / ", niterations, ",   ", "number of particles: ", len(state.particles))
        for particle in state.particles:
            offsets.append([particle.x, particle.y])
            sizes.append(10919 * particle.R**2)
        scat.set_offsets(offsets)
        scat.set_sizes(sizes)
        for _ in range(step_size):
            state.Step()
        return scat,

    myAnimation = FuncAnimation(fig, update, frames = niterations, interval = 10,
                                blit=True, repeat=False)
    myAnimation.save('pauze.gif', writer='imagemagick', fps=30)

if __name__ == "__main__":
    animation(live=True, make_GIF=False)
