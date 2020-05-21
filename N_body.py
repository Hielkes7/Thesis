import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
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
mass_sun = 1.989e30 # kg
G = 6.67408e-11 # m^3 kg^-1 s^-2, the gravitational constant


class State():
    """
    This class is the current state of a set up involving a sun and x amount
    of clumps.
    """
    def __init__(self, toggle_3D, amount_clumps, dt, boundary_box, \
                 clump_radius, clump_distribution, max_velocity_fraction, \
                 curl_fraction, cloud_mass):
        self.amount_clumps = amount_clumps
        self.clumps = []
        self.boundary_box = boundary_box
        self.time = 0
        self.dt = dt
        self.begin_time = time.time()
        self.star = None
        self.CM = None
        self.toggle_3D = toggle_3D

        # toggle variables, they get toggled on somewhere else
        self.gravity_star_on = False
        self.gravity_clumps_on = False
        self.radiation_pressure_on = False
        self.gas_pressure_on = False

        if clump_distribution == "constant":
            self.constant_clump_distribution = True
            self.power_law_clump_distribution = False
        if clump_distribution == "power_law":
            self.constant_clump_distribution = False
            self.power_law_clump_distribution = True

        # initiate all positions of the clumps
        # random.seed(0)
        for _ in range(self.amount_clumps):
            r_max = boundary_box[1] * 0.8
            if self.constant_clump_distribution:
                r = random.random() * r_max
            if self.power_law_clump_distribution:
                r = 1/ random.random()**2 * r_max / 100

            # direction position according to physics standard. phi is angle
            # with the x-axis and [0,2pi], theta is angle with z-axis [0,pi]
            phi = random.random() * 2 * np.pi
            if self.toggle_3D:
                theta = random.random() * np.pi
            else:
                theta = 0.5 * np.pi

            x = r * np.sin(theta) * np.cos(phi) # m
            y = r * np.sin(theta) * np.sin(phi) # m
            z = r * np.cos(theta)

            # the initial velocity will be calculated later on in the code
            vx = 0
            vy = 0
            vz = 0

            R = clump_radius # m
            m = Mass_clump(R)
            clump = Object(x, y, z, vx, vy, vz, m, R)
            self.clumps.append(clump)

        # determine centre of mass (probably close to 0,0)
        x_sum = 0
        y_sum = 0
        z_sum = 0
        mass_sum = 0
        for clump in self.clumps:
            mass_sum += clump.m
            x_sum += clump.m * clump.x
            y_sum += clump.m * clump.y
            z_sum += clump.m * clump.z

        x_CM = x_sum / mass_sum
        y_CM = y_sum / mass_sum
        z_CM = z_sum / mass_sum
        self.CM = [x_CM, y_CM, z_CM]

        # initiate all starting velocities of the clumps
        for clump in self.clumps:
            max_velocity = max_velocity_fraction * self.Orbital_speed(clump)

            # random velocity in random direction
            phi = random.random() * 2 * np.pi
            if self.toggle_3D:
                theta = random.random() * np.pi
            else:
                theta = 0.5 * np.pi

            v_random = random.random() * max_velocity
            vx_random = v_random * np.sin(theta) * np.cos(phi)
            vy_random = v_random * np.sin(theta) * np.sin(phi)
            vz_random = v_random * np.cos(theta)

            # chosen velocity in fixed direction to generate curl to the cloud
            dx = clump.x - self.CM[0]
            dy = clump.y - self.CM[1]
            dz = clump.z - self.CM[2]
            if dx > 0:
                phi = np.arctan(dy / dx)
            if dx < 0:
                phi = np.pi + np.arctan(dy / dx)
            vx_curl = -np.sin(phi) * max_velocity
            vy_curl =  np.cos(phi) * max_velocity

            # initial velocity is combination of chosen and random velocity
            # with a ratio set by the variable "curl_fraction"
            clump.vx = curl_fraction * vx_curl + (1 - curl_fraction) * vx_random
            clump.vy = curl_fraction * vy_curl + (1 - curl_fraction) * vy_random
            clump.vz = vz_random


    def Initiate_star(self, R_star, M_star, QH):
        """
        This function initializes a star in the middle of the 2D plot
        """
        self.QH = QH # amount of ionizing photons
        x, y, z, vx, vy, vz = 0, 0, 0, 0, 0, 0
        star = Object(x, y, z, vx, vy, vz, M_star, R_star)
        self.star = star

    def Step(self):
        """
        This function executes one timestep of length dt using Eulers method.
        """
        # reset accelerations of clumps from previous iterations
        for clump in self.clumps:
            clump.ax = 0
            clump.ay = 0
            clump.az = 0

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
            clump.vz += clump.az * self.dt

            clump.x += clump.vx * self.dt
            clump.y += clump.vy * self.dt
            clump.z += clump.vz * self.dt

            # if clumps have escaped the cloud and are beyond view, remove them
            if abs(clump.x) > 2 * self.boundary_box[1]:
                self.clumps.remove(clump)
            elif abs(clump.y) > 2 * self.boundary_box[1]:
                self.clumps.remove(clump)
            elif abs(clump.z) > 2 * self.boundary_box[1]:
                self.clumps.remove(clump)

        # update time
        self.time += self.dt

    def Plot(self):
        """
        This function makes a plot of the current state of the set up
        """
        fig = plt.figure()
        fig.set_size_inches(10, 10) # 10 inches wide and long

        if self.toggle_3D:
            ax = fig.add_subplot(111, projection='3d')

            if self.star:
                ax.scatter(self.star.x, self.star.y, self.star.z, s= 8e4 * \
                           self.star.R**2 / self.boundary_box[1]**2, c='red')
            for clump in self.clumps:
                ax.scatter(clump.x, clump.y, clump.z, s= np.pi * 1e5 * clump.R**2\
                           / self.boundary_box[1]**2, c='red')

            ax.set_xlim(self.boundary_box)
            ax.set_ylim(self.boundary_box)
            ax.set_zlim(self.boundary_box)
            ax.set_xlabel('x axis (meter)')
            ax.set_ylabel('y axis (meter)')
            ax.set_zlabel('z axis (meter)')

        # plot of 2D state
        else:
            ax = fig.add_subplot(111)
            if self.star:
                ax.scatter(self.star.x, self.star.y, s= np.pi * 1e5 * \
                           self.star.R**2 / self.boundary_box[1]**2, c='red')
            for clump in self.clumps:
                ax.scatter(clump.x, clump.y, s= np.pi * 1e5 * clump.R**2\
                           / self.boundary_box[1]**2, c='red')
            plt.xlim(self.boundary_box)
            plt.ylim(self.boundary_box)

            ax.set_xlabel('x axis (meter)')
            ax.set_ylabel('y axis (meter)')
        plt.grid()
        plt.show()

    def Radiation_pressure(self):
        """
        This function calculates the acceleration of the clumps due to the
        radiation pressure by the star. TODO: The formula used to calculate the
        radiation pressure is incorrect, find and apply the actual formula
        """
        for clump in self.clumps:
            dr, dx, dy, dz = self.Distance(clump, self.star)
            # a = 6.25e-50 * self.QH / dr**3
            a = 5e30 / dr**3
            clump.ax += -a * dx / dr
            clump.ay += -a * dy / dr
            clump.az += -a * dz / dr

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
                dr, dx, dy, dz = self.Distance(self.clumps[i], self.clumps[j])
                a1 = G * self.clumps[j].m / dr**2
                a2 = G * self.clumps[i].m / dr**2
                self.clumps[i].ax += -a1 * dx / dr
                self.clumps[i].ay += -a1 * dy / dr
                self.clumps[i].az += -a1 * dz / dr

                self.clumps[j].ax += a2 * dx / dr
                self.clumps[j].ay += a2 * dy / dr
                self.clumps[j].az += a2 * dz / dr
                j += 1
            i += 1

    def Gravity_star(self):
        """
        This function calculate the acceleration of all clumps due to the
        gravitational pull of the star
        """
        if not self.star:
            raise Exception("Error, can't calculate gravity of star. There is no star")

        for clump in self.clumps:
            dr, dx, dy, dz = self.Distance(clump, self.star)

            # absorption of clump by star
            if dr < self.star.R + clump.R:
                self.clumps.remove(clump)

            # if no absorption, calculate acceleration due to gravity
            else:
                a = G * self.star.m / dr**2
                clump.ax += a * dx / dr
                clump.ay += a * dy / dr
                clump.az += a * dz / dr

    def Gravity_clumps(self):
        """
        This function calculate the acceleration of all clumps due to the
        gravitational pull of the other clumps
        """
        i = 0
        while i < len(self.clumps):
            j = i + 1
            while j < len(self.clumps):
                dr, dx, dy, dz = self.Distance(self.clumps[i], self.clumps[j])
                if not self.Collision(self.clumps[i], self.clumps[j], dr):
                    a1 = G * self.clumps[j].m / dr**2
                    a2 = G * self.clumps[i].m / dr**2
                    self.clumps[i].ax += a1 * dx / dr
                    self.clumps[i].ay += a1 * dy / dr
                    self.clumps[i].az += a1 * dz / dr

                    self.clumps[j].ax += -a2 * dx / dr
                    self.clumps[j].ay += -a2 * dy / dr
                    self.clumps[j].az += -a2 * dz / dr
                j += 1
            i += 1

    def Orbital_speed(self, clump):
        """
        This function returns the velocity needed for a clump to remain in stable
        orbit. This is calculated by ignorning all the gravity of clumps further
        away of the CM than the clump being looked at.
        """
        distance1 = np.sqrt((clump.x - self.CM[0])**2 + \
                            (clump.y - self.CM[1])**2 + \
                            (clump.z - self.CM[2])**2)

        # find total mass of all clumps closer to the CM than the current clump
        sum_mass = 0
        for other_clump in self.clumps:
            distance2 = np.sqrt((other_clump.x - self.CM[0])**2 + \
                                (other_clump.y - self.CM[1])**2 +
                                (other_clump.z - self.CM[2])**2)
            if distance2 < distance1:
                sum_mass += other_clump.m

        # treat all the clumps closer to the CM as a point source in the CM
        orbital_speed = np.sqrt(G * sum_mass / distance1)
        return orbital_speed

    def Distance(self, clump1, clump2):
        """
        This function returns the distances between the two given clumps
        """
        dx = clump2.x - clump1.x
        dy = clump2.y - clump1.y
        dz = clump2.z - clump1.z
        dr = (dx**2 + dy**2 + dz**2)**0.5
        return dr, dx, dy, dz

    def Collision(self, clump1, clump2, dr):
        """
        This function checks if there have been clumps which collided, it will
        then create a new clump of combined mass, momentum and new corresponding
        size. "clump1" will be updated, "clump2" will be removed.
        """
        # the merge factor states how much two clumps have to overlap before
        # they merge
        merge_factor = 0.2
        if dr < merge_factor * clump1.R + clump2.R:
            # find centre of mass
            clump1.x = (clump1.x *clump1.m + clump2.x * clump2.m)\
                          / (clump1.m + clump2.m)
            clump1.y = (clump1.y *clump1.m + clump2.y * clump2.m)\
                          / (clump1.m + clump2.m)
            clump1.z = (clump1.z *clump1.m + clump2.z * clump2.m)\
                          / (clump1.m + clump2.m)

            # find new velocity with conservation of impuls
            clump1.vx = (clump1.vx *clump1.m + clump2.vx * clump2.m)\
                           / (clump1.m + clump2.m)
            clump1.vy = (clump1.vy *clump1.m + clump2.vy * clump2.m)\
                           / (clump1.m + clump2.m)
            clump1.vz = (clump1.vz *clump1.m + clump2.vz * clump2.m)\
                           / (clump1.m + clump2.m)

            clump1.m = clump1.m + clump2.m
            clump1.R = Radius_clump(clump1.m)
            self.clumps.remove(clump2)
            return True
        return False

    def __str__(self):
        if toggle_3D:
            return f"{self.amount_clumps} clumps in a {self.boundary_box[1]}x\
                   {self.boundary_box[1]}x{self.boundary_box[1]} box"
        else:
            return f"{self.amount_clumps} clumps in a {self.boundary_box[1]}x\
                   {self.boundary_box[1]} box"


class Object():
    """
    This class contains information about a clump
    """
    def __init__(self, x, y, z, vx, vy, vz, m, R):
      self.x = x    # position x coordinate
      self.y = y    # position y coordinate
      self.z = z    # position z coordinate
      self.vx = vx      # velocity in x direction
      self.vy = vy      # velocity in y direction
      self.vz = vz      # velocity in z direction
      self.ax = 0       # acceleration in x direction
      self.ay = 0       # acceleration in y direction
      self.az = 0       # acceleration in z direction
      self.m = m
      self.R = R

    def __str__(self):
      return f"A clump of mass {self.m} and radius \
             {self.R} at position ({self.x}, {self.y})"

def Radius_clump(mass):
    """
    This function calculates the radius of clumps depending on their mass.
    I derived this formula of excisting data of mass/radius ratios. See
    the file "radius_mass_ratio_clumps.pdf" on my github:
    https://github.com/Hielkes7/Thesis

    R = (M**0.426)/55.55
    R in pc and M in solar masses
    """
    radius = pc / 55.55 * (mass/mass_sun)**0.426
    return radius

def Mass_clump(radius):
    """
    This function calculates the mass of clumps depending on their radius.
    I derived this formula of excisting data of mass/radius ratios. See
    the file "radius_mass_ratio_clumps.pdf" on my github:
    https://github.com/Hielkes7/Thesis

    M = 12590 * R**2.35
    M in solar masses and R in pc
    """
    mass = 12590 * mass_sun * (radius/pc)**2.35
    return mass

def animation(animate_live, make_GIF, state, amount_of_frames, niterations, size_box):
    """
    This function animates evolution of the set up. There is an option for live
    animation or the making of GIF
    """
    # animation settings
    boundary_box = -size_box/2, size_box/2
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

    title = ax.text(0.5, 1.02, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha="center")

    def update(frame):
        if make_GIF and frame != 0:
            # information feedback to estimate duration of animation
            current_sec = int(time.time() - state.begin_time)
            current_min = int(current_sec / 60)
            current_hour = int(current_min / 60)

            total_sec = int(current_sec * (amount_of_frames + 30) / frame)
            total_min = int(total_sec / 60)
            total_hours = int(total_min / 60)

            hours_left = total_hours - current_hour
            min_left = total_min - current_min
            sec_left = total_sec - current_sec

            print("Frame %d / %d       Total time: %dh%dm%s     Time left:  %dh%dm%s    Sec left: %d" \
                  %(frame - 30, amount_of_frames, total_hours, total_min%60, \
                  total_sec%60, hours_left, min_left%60, sec_left%60, \
                  sec_left), end="\r")

        offsets = []
        sizes = []
        title.set_text(u"{} / {} iterations - {} Myr".format(frame*step_size,\
                       niterations, round(state.time / Myr, 1)))

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

        # adding a second of stationary footage to make watching easier
        if frame > 30:
            for _ in range(step_size):
                state.Step()
        return scat,title,

    myAnimation = FuncAnimation(fig, update, frames = amount_of_frames + 30, \
                                interval = 10, repeat=False)
    ax.set_xlim(boundary_box[0], boundary_box[1])
    ax.set_ylim(boundary_box[0], boundary_box[1])
    ax.set_xlabel('Distance (meters)')
    ax.set_ylabel('Distance (meters)')
    if animate_live:
        plt.show()
    if make_GIF:
        myAnimation.save('RENAME_THIS_GIF.gif', writer='imagemagick', fps=30)

def set_up():
    """
    This function contains all the input values. The user adjusts them.
    """
    # simulation settings
    time_frame =  10 * Myr # seconds, about the age of our solar system
    niterations = int(3000)
    size_box = 20 * pc # diameter of orbit of pluto
    toggle_3D = False


    # animation settings
    boundary_box = -size_box/2, size_box/2
    amount_of_frames = int(300)
    dt = time_frame / niterations # s

    # star settings
    # minimum visable radius is size_box / 1000
    R_star = 2e17 # radius sun displayed in anmiation, not in scale
    M_star = 1.989e30# mass sun in kg
    QH = 1e45 # photon per second emitted

    # clump settings
    amount_clumps = 50
    cloud_mass = 3400 * mass_sun # obtained from the data Sam gave me, not containing background gas yet
    clump_mass = cloud_mass / amount_clumps
    clump_radius = Radius_clump(clump_mass)
    max_velocity_fraction = 1
    curl_fraction = 1

    # choose one
    clump_distribution = "constant"
    # clump_distribution = "power_law"

    # initializing begin state
    state = State(toggle_3D, amount_clumps, dt, boundary_box, clump_radius, \
                  clump_distribution, max_velocity_fraction, curl_fraction, \
                  cloud_mass)
    # state.Initiate_star(R_star, M_star, QH)

    # toggle force parameters
    # state.gravity_star_on = True
    state.gravity_clumps_on = True
    # state.radiation_pressure_on = True
    # state.gas_pressure_on = True
    # TODO state.clump_evaportation_on = True
    # TODO state.stellar_wind_on = True

    # Choose either one, they can't both be True
    make_GIF = False # you can only make GIF's on anaconda prompt after installing FFmpeg: conda install -c menpo ffmpeg
    animate_live = True
    animation(animate_live, make_GIF, state, amount_of_frames, niterations, \
              size_box)

def test_3D():
    """
    This function contains all the input values. The user adjusts them.
    """
    # simulation settings
    time_frame =  10 * Myr # seconds, about the age of our solar system
    niterations = int(3000)
    size_box = 20 * pc # diameter of orbit of pluto
    toggle_3D = True


    # animation settings
    boundary_box = -size_box/2, size_box/2
    amount_of_frames = int(300)
    dt = time_frame / niterations # s

    # star settings
    # minimum visable radius is size_box / 1000
    R_star = 1e17 # radius sun displayed in anmiation, not in scale
    clump_radius = 5e14
    M_star = 1.989e30# mass sun in kg
    QH = 1e45 # photon per second emitted

    # clump settings
    amount_clumps = 10
    cloud_mass = 3400 * mass_sun # obtained from the data Sam gave me, not containing background gas yet
    clump_mass = cloud_mass / amount_clumps
    clump_radius = Radius_clump(clump_mass)
    max_velocity_fraction = 1
    curl_fraction = 1

    # choose one
    clump_distribution = "constant"
    # clump_distribution = "power_law"

    # initializing begin state
    state = State(toggle_3D, amount_clumps, dt, boundary_box, clump_radius, \
                  clump_distribution, max_velocity_fraction, curl_fraction, \
                  cloud_mass)
    # state.Initiate_star(R_star, M_star, QH)

    state.gravity_clumps_on = True

    state.Plot()


if __name__ == "__main__":
    test_3D()
    # set_up()
