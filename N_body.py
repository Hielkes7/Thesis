import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import random
import matplotlib.pyplot as plt
import time
import collections
from matplotlib.animation import FuncAnimation
from all_data_clumps import data_clumps

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

# some numbers to get a better feeling for the situation
mass_density_clump = 2.877e-16 # kg/m^3 (clump of M= 3400 * M_sun)


class State():
    """
    This class is the current state of a set up involving a sun and x amount
    of clumps.
    """
    def __init__(self, toggle_3D, amount_clumps, dt, boundary_box,clump_distribution, max_velocity_fraction, curl_fraction, cloud_mass, initial_clump_mass, use_Sams_data, random_movement_fraction):
        self.amount_clumps = amount_clumps
        self.clumps = []
        self.boundary_box = boundary_box
        self.time = 0
        self.dt = dt
        self.begin_time = time.time()
        self.star = None
        self.CM = None
        self.toggle_3D = toggle_3D
        self.initial_clump_mass = initial_clump_mass
        self.collision_data = []

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
        random.seed(24352354)
        if use_Sams_data:
            for clump_name in data_clumps:

                # place first clump as starter
                m = data_clumps[clump_name]['mass'] * mass_sun
                R = data_clumps[clump_name]['radius'] * pc
                d = data_clumps[clump_name]['distance'] * pc

                # find angle which puts the CM closest to (0,0,0)
                shortest_distance_to_origin = 9e99 # dummy large number
                for phi in np.arange(0, 2*np.pi, 0.1):
                    theta = 0.5 * np.pi # TODO, make 3D, right now z=0
                    x = d * np.sin(theta) * np.cos(phi) # m
                    y = d * np.sin(theta) * np.sin(phi) # m
                    z = d * np.cos(theta)
                    clump = Object(x, y, z, 0, 0, 0, m, R)
                    self.clumps.append(clump)

                    # determine CM and its distance to (0,0,0)
                    self.Find_CM()
                    distance_to_origin = np.sqrt(self.CM[0]**2 +\
                                                 self.CM[1]**2 +\
                                                 self.CM[2]**2)

                    if distance_to_origin < shortest_distance_to_origin:
                        shortest_distance_to_origin = distance_to_origin
                        best_phi = phi

                    # delete temporary location to possibly find better one
                    self.clumps.remove(clump)

                # add best location clump
                x = d * np.sin(theta) * np.cos(best_phi) # m
                y = d * np.sin(theta) * np.sin(best_phi) # m
                z = d * np.cos(theta)
                clump = Object(x, y, z, 0, 0, 0, m, R)
                self.clumps.append(clump)

        else:
            for _ in range(self.amount_clumps):
                d_max = boundary_box[1] * 0.8
                if self.constant_clump_distribution:
                    d = random.random() * d_max
                if self.power_law_clump_distribution:
                    d = 1/ random.random()**2 * d_max / 100

                # direction position according to physics standard. phi is angle
                # with the x-axis and [0,2pi], theta is angle with z-axis [0,pi]
                phi = random.random() * 2 * np.pi
                if self.toggle_3D:
                    theta = random.random() * np.pi
                else:
                    theta = 0.5 * np.pi

                x = d * np.sin(theta) * np.cos(phi) # m
                y = d * np.sin(theta) * np.sin(phi) # m
                z = d * np.cos(theta)

                # the initial velocity will be calculated later on in the code
                vx = 0
                vy = 0
                vz = 0

                m = initial_clump_mass
                R = self.Radius_clump(m)
                clump = Object(x, y, z, vx, vy, vz, m, R)
                self.clumps.append(clump)

            # determine centre of mass (probably close to 0,0)
            self.Find_CM()

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
            clump.vx = curl_fraction * vx_curl + random_movement_fraction * vx_random
            clump.vy = curl_fraction * vy_curl + random_movement_fraction * vy_random
            clump.vz = vz_random

    def Initiate_star(self, R_star, M_star):
        """
        This function initializes a star in the middle of the 2D plot
        """
        x, y, z, vx, vy, vz = 0, 0, 0, 0, 0, 0
        star = Object(x, y, z, vx, vy, vz, M_star, R_star)
        self.star = star

    def Step(self):
        """
        This function executes one timestep of length dt using Eulers method.
        """
        # reset accelerations of clumps from previous iterations
        self.star.ax = 0
        self.star.ay = 0
        self.star.az = 0
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

        # determine new CM of cloud
        self.Find_CM()

        # calcualte new position of star (TODO, keep everyting relative to the star and have it always at (0,0,0))
        self.star.vx += self.star.ax * self.dt
        self.star.vy += self.star.ay * self.dt
        self.star.vz += self.star.az * self.dt

        self.star.x += self.star.vx * self.dt
        self.star.y += self.star.vy * self.dt
        self.star.z += self.star.vz * self.dt

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

        # creating ticks on axis
        amount_of_pc = int(self.boundary_box[1] / pc) * 2 + 1
        max_amount_ticks = 21
        factor_pc = int(amount_of_pc / max_amount_ticks) + 1
        amount_of_ticks = int(amount_of_pc / factor_pc) + 1
        middle_tick = int(amount_of_ticks / 2) # should be +1 but since python starts counting at 0, i is the (i+1)th item
        distance_values = []
        axis_labels = []
        for i in range(amount_of_ticks):
            axis_labels.append((i - middle_tick) * factor_pc)
            distance_values.append((i - middle_tick) * factor_pc * pc)

        # if the simulation is in 3D
        if self.toggle_3D:
            ax = fig.add_subplot(111, projection='3d')

            if self.star:
                ax.scatter(self.star.x, self.star.y, self.star.z, s= 8e4 * \
                           self.star.R**2 / self.boundary_box[1]**2, c='red')
            for clump in self.clumps:
                ax.scatter(clump.x, clump.y, clump.z, s= np.pi * 1e5 * clump.R**2\
                           / self.boundary_box[1]**2, c='red')


            ax.set_zlim(self.boundary_box)
            ax.set_zlabel('Distance (pc)')
            ax.set_zticks(distance_values)
            ax.set_zticklabels(axis_labels)

        # plot of 2D state
        else:
            ax = fig.add_subplot(111)
            if self.star:
                ax.scatter(self.star.x, self.star.y, s= np.pi * 1e5 * \
                           self.star.R**2 / self.boundary_box[1]**2, c='red')
            for clump in self.clumps:
                ax.scatter(clump.x, clump.y, s= np.pi * 1e5 * clump.R**2\
                           / self.boundary_box[1]**2, c='red')

        # settings that apply for both 2D and 3D
        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('Distance (pc)')

        ax.set_xticks(distance_values)
        ax.set_xticklabels(axis_labels)
        ax.set_yticks(distance_values)
        ax.set_yticklabels(axis_labels)

        ax.set_xlim(self.boundary_box)
        ax.set_ylim(self.boundary_box)

        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('Distance (pc)')
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
            if not self.Collision(self.star, clump, dr):
                a_clump = G * self.star.m / dr**2
                clump.ax += a_clump * dx / dr
                clump.ay += a_clump * dy / dr
                clump.az += a_clump * dz / dr

                a_star = G * clump.m / dr**2
                self.star.ax += -a_star * dx / dr
                self.star.ay += -a_star * dy / dr
                self.star.az += -a_star * dz / dr

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
                                (other_clump.y - self.CM[1])**2 + \
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
        merge_factor = 0.6
        if dr < merge_factor * (clump1.R + clump2.R):

            # impact velocity
            abs_velocity1 = np.sqrt(clump1.vx**2 + clump1.vy**2 + clump1.vz**2)
            abs_velocity2 = np.sqrt(clump2.vx**2 + clump2.vy**2 + clump2.vz**2)
            product_abs_velocity = abs_velocity1 * abs_velocity2

            inproduct_vectors = clump1.vx * clump2.vx + \
                                clump1.vy * clump2.vy + \
                                clump1.vz * clump2.vz

            # angle between the velocity vectors
            theta = np.arccos(inproduct_vectors / product_abs_velocity)
            impact_velocity1 = abs_velocity1 * np.sin(theta)
            impact_velocity2 = abs_velocity2 * np.sin(theta)
            tot_impact_velocity = impact_velocity1 + impact_velocity2


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

            # distance from the collision to the CM of the cloud
            CM_x = self.CM[0]
            CM_y = self.CM[1]
            CM_z = self.CM[2]
            distance = np.sqrt((CM_x - clump1.x)**2 + \
                               (CM_y - clump1.y)**2 + \
                               (CM_z - clump1.z)**2)

            # saving collision data
            collision = {
               "impact_velocity": tot_impact_velocity,
               "time": self.time,
               "clump_masses": [clump1.m, clump2.m],
               "distance_to_CM": distance
            }
            self.collision_data.append(collision)

            clump1.m = clump1.m + clump2.m
            clump1.R = self.Radius_clump(clump1.m)
            self.clumps.remove(clump2)
            return True

        return False

    def Find_CM(self):
        """
        This function finds the centre of mass of the cloud
        """
        # CM_x = (x1 * m1 + x2 + m2) / (m1 + m2) = numerator / denominator
        numerator_x = 0
        numerator_y = 0
        numerator_z = 0
        denominator = 0
        if self.star:
            numerator_x += self.star.x * self.star.m
            numerator_y += self.star.y * self.star.m
            numerator_z += self.star.z * self.star.m
            denominator += self.star.m
        for clump in self.clumps:
            numerator_x += clump.x * clump.m
            numerator_y += clump.y * clump.m
            numerator_z += clump.z * clump.m
            denominator += clump.m
        if self.star or self.clumps:
            self.CM = [numerator_x / denominator, \
                       numerator_y / denominator, \
                       numerator_z / denominator]
            return True

    def Radius_clump(self, mass):
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

    def Mass_clump(self, radius):
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

def animation(make_GIF, state, amount_of_frames, niterations, size_box, animate_hist, animate_scat, hist_axis_fixed, animate_CM, animate_HII_region, file_name):
    """
    This function animates evolution of the set up. There is an option for live
    animation or the making of GIF
    """
    # animation settings
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

    # combined plot of the scatter animation and histogram of mass specturm
    if animate_hist and animate_scat:
        fig, (ax_scat, ax_hist) = plt.subplots(1, 2)
        fig.set_size_inches(17.6, 8) # inches wide and long

    # just the animation of the scatter plots of clumps
    elif animate_scat:
        fig, ax_scat = plt.subplots(1, 1)
        fig.set_size_inches(10, 10) # 10 inches wide and long

    # just the histogram of the mass specturm
    elif animate_hist:
        fig, ax_hist = plt.subplots(1, 1)
        fig.set_size_inches(10, 10) # 10 inches wide and long

    if animate_scat:
        ax_scat.grid(True)

        # animate HII-region
        if animate_HII_region:
            back_ground_gas = ax_scat.scatter(0, 0, s=1.24e6, \
                              label = "Background gas", facecolor = "lightsalmon")

            # radius_HII = get_HII_radius

            HII_region = ax_scat.scatter(state.star.x, state.star.y, \
                         s=1.24e5, \
                         label = "HII region", facecolor = "white")

        scat = ax_scat.scatter(x, y, label = "Gas clumps", facecolor = "red")

        title_scat = ax_scat.text(0.5, 1.02, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                        transform=ax_scat.transAxes, ha="center")

        # creating ticks on axis
        amount_of_pc = int(state.boundary_box[1] / pc) * 2 + 1
        max_amount_ticks = 21
        factor_pc = int(amount_of_pc / max_amount_ticks) + 1
        amount_of_ticks = int(amount_of_pc / factor_pc) + 1
        middle_tick = int(amount_of_ticks / 2) # should be +1 but since python starts counting at 0, i is the (i+1)th item
        distance_values = []
        axis_labels = []
        for i in range(amount_of_ticks):
            axis_labels.append((i - middle_tick) * factor_pc)
            distance_values.append((i - middle_tick) * factor_pc * pc)

        ax_scat.set_xlabel('Distance (pc)')
        ax_scat.set_ylabel('Distance (pc)')

        ax_scat.set_xticks(distance_values)
        ax_scat.set_xticklabels(axis_labels)
        ax_scat.set_yticks(distance_values)
        ax_scat.set_yticklabels(axis_labels)

        ax_scat.set_xlim(state.boundary_box)
        ax_scat.set_ylim(state.boundary_box)

        def update_scat(frame):
            if make_GIF and frame != 0:
                # information feedback to estimate duration of animation
                current_sec = int(time.time() - state.begin_time)
                current_min = int(current_sec / 60)
                current_hour = int(current_min / 60)

                total_sec = int(current_sec * amount_of_frames / frame)
                total_min = int(total_sec / 60)
                total_hours = int(total_min / 60)

                hours_left = total_hours - current_hour
                min_left = total_min - current_min
                sec_left = total_sec - current_sec

                print("Frame %d / %d       Total time: %dh%dm%s     Time left:  %dh%dm%s    Sec left: %d" \
                      %(frame, amount_of_frames, total_hours, total_min%60, \
                      total_sec%60, hours_left, min_left%60, sec_left%60, \
                      sec_left), end="\r")

            offsets = []
            sizes = []
            title_scat.set_text(u"{} / {} iterations - {} Myr".format(frame*step_size,\
                           niterations, round(state.time / Myr, 1)))

            # animate star
            if state.star:
                scat_star = ax_scat.scatter(state.star.x, state.star.y, label = "Star", facecolor = "blue")

            # animate clumps
            for clump in state.clumps:
                offsets.append([clump.x, clump.y])
                sizes.append(1.24e6 * clump.R**2 * size_box**(-2))
            scat.set_offsets(offsets)
            scat.set_sizes(sizes)

            # centre of mass
            if animate_CM:
                scat_CM = ax_scat.scatter(state.CM[0], state.CM[1], label = "Centre of Mass", facecolor = "green")

            print("particles: %d / %d" %(len(state.clumps), state.amount_clumps))
            print("Time: %.2f Myr" %round(state.time / Myr, 2))
            print()

            # each frame has "step_size" iterations done
            for _ in range(step_size):
                state.Step()
            if animate_CM:
                return back_ground_gas, HII_region, scat, title_scat, scat_CM, scat_star
            else:
                return back_ground_gas, HII_region, scat, title_scat

        # blit=True makes it run alot faster but the title gets removed
        if not make_GIF:
            myAnimation_scat = FuncAnimation(fig, update_scat, \
                               frames = amount_of_frames, \
                               interval = 10, repeat=False,
                               blit=True)
        else:
            myAnimation_scat = FuncAnimation(fig, update_scat, \
                               frames = amount_of_frames, \
                               interval = 10, repeat=False,
                               blit=True)

    # the histogram part
    if animate_hist:
        title_hist = ax_hist.text(0.5, 1.02, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                        transform=ax_hist.transAxes, ha="center")

        def update_hist(frame):
            title_hist.set_text("Mass spectrum of clumps")

            # make a list with all masses of all clumps
            all_masses = []
            for clump in state.clumps:
                all_masses.append(clump.m)

            # we want no more than 20 ticks
            axis_ticks = []
            size_ticks = int(state.amount_clumps / 21) + 1
            amount_of_ticks = int(state.amount_clumps / size_ticks)
            for i in range(amount_of_ticks + 1):
                axis_ticks.append(i * size_ticks)

            # creating bins list
            bins = []
            for i in range(state.amount_clumps + 1):
                bins.append(state.initial_clump_mass * (i + 0.5))

            mass_values = []
            axis_labels = []
            amount_ticks_x_axis = 10
            size_ticks = state.amount_clumps / amount_ticks_x_axis
            for i in range(1, amount_ticks_x_axis + 1):
                mass_values.append(state.initial_clump_mass * i * size_ticks)
                axis_labels.append(int(i * size_ticks))

            ax_hist.cla()
            ax_hist.hist(all_masses, bins=bins, rwidth=0.75)
            ax_hist.set_xticks(mass_values)
            if hist_axis_fixed:
                ax_hist.set_yticks(axis_ticks)
            ax_hist.set_xticklabels(axis_labels)
            ax_hist.set_xlabel('Mass (initial clump mass)')
            ax_hist.set_ylabel('Frequency')
            plt.title("Mass spectrum after %d iterations" %(frame * step_size))

            # each frame has "step_size" iterations done
            if not animate_scat:
                for _ in range(step_size):
                    state.Step()
            return title_hist,

        # blit=True makes it run alot faster but the title gets removed
        if not make_GIF:
            myAnimation_hist = FuncAnimation(fig, update_hist, \
                               frames=amount_of_frames, \
                               interval=10, repeat=False, blit=True)
        else:
            myAnimation_hist = FuncAnimation(fig, update_hist, \
                               frames=amount_of_frames, \
                               interval=10, repeat=False, blit=True)

    if make_GIF:
        if animate_scat:
            myAnimation_scat.save('RENAME_THIS_GIF.gif', writer='imagemagick', fps=30)
        if animate_hist:
            myAnimation_hist.save('RENAME_THIS_GIF.gif', writer='imagemagick', fps=30)
    else:
        plt.show()

def set_up():
    """
    This function contains all the input values. The user adjusts them.
    """
    # simulation settings
    time_frame =  20 * Myr # seconds, about the age of our solar system
    niterations = int(500 * time_frame / Myr)
    size_box = 20 * pc # diameter of orbit of pluto
    toggle_3D = False

    # animation settings
    boundary_box = -size_box/2, size_box/2
    amount_of_frames = int(niterations / 10)
    dt = time_frame / niterations # s
    animate_HII_region = True
    file_name = "n0=1000, rmax=10pc, cool_off, grav_off, T0=10"

    # star settings
    # minimum visable radius is size_box / 1000
    R_star = 40 * R_sun # radius sun displayed in anmiation, not in scale
    M_star = 35 * mass_sun
    QH = 1e45 # photon per second emitted

    # clump settings
    use_Sams_data = False
    amount_clumps = 150
    cloud_mass = 3400 * mass_sun # obtained from the data Sam gave me, not containing background gas yet
    initial_clump_mass = cloud_mass / amount_clumps
    max_velocity_fraction = 0.8
    curl_fraction = 1
    random_movement_fraction = 1

    # choose one
    clump_distribution = "constant"
    # clump_distribution = "power_law"

    # initializing begin state
    state = State(toggle_3D, amount_clumps, dt, boundary_box, \
                  clump_distribution, \
                  max_velocity_fraction, curl_fraction, cloud_mass, \
                  initial_clump_mass, use_Sams_data, random_movement_fraction)
    state.Initiate_star(R_star, M_star)
    # state.Plot()

    # toggle force parameters
    state.gravity_star_on = True
    state.gravity_clumps_on = True
    state.radiation_pressure_on = False
    state.gas_pressure_on = False
    state.clump_evaportation_on = False # TODO
    state.stellar_wind_on = False # TODO

    # Choose either one, they can't both be True
    make_GIF = False # you can only make GIF's on anaconda prompt after installing FFmpeg: conda install -c menpo ffmpeg
    animate_scat = True
    animate_hist = False
    hist_axis_fixed = False
    animate_CM = True
    animation(make_GIF, state, amount_of_frames, \
              niterations, size_box, animate_hist, animate_scat, \
              hist_axis_fixed, animate_CM, animate_HII_region, file_name)

if __name__ == "__main__":
    set_up()
