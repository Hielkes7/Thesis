import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import random
import matplotlib.pyplot as plt
import time
import collections
from matplotlib.animation import FuncAnimation
# from all_data_clumps import data_clumps
import pandas as pd
import os

my_path = os.path.abspath(__file__)[:-len("N_body.py")]


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
m_sun = 1.989e30 # kg
R_sun = 6.9634e8 # meters
G = 6.67408e-11 # m^3 kg^-1 s^-2, the gravitational constant
u = 1.660539e-27 # kg, atomic mass unit
m_H = 1.00784*u # kg, mass hydrogen
n_BGG = 1e8 # particles per m^3
rho_BGG = 1.7907e-19 # kg m^-3, mass density background gas
R_edge_cloud = 2.46854e17 # m, 8pc
m_tot_clumps = 3400 * m_sun # total mass of all clumps from Sam's data


class State():
    """
    This class is the current state of a set up involving a sun and x amount
    of clumps.
    """
    def __init__(self, toggle_3D, amount_clumps, dt, boundary_box, clump_distribution, max_velocity_fraction, curl_fraction, cloud_mass, initial_clump_mass, use_Sams_data, random_movement_fraction):
        self.amount_clumps = amount_clumps
        self.clumps = []
        self.boundary_box = boundary_box
        self.size_box = 2 * boundary_box[1]
        self.time = 0
        self.dt = dt
        self.begin_time = time.time()
        self.star = None
        self.CM = None
        self.toggle_3D = toggle_3D
        self.initial_clump_mass = initial_clump_mass
        self.collision_data = []
        self.background_gas = False
        self.weltgeist_data = None
        self.use_weltgeist_dummy_data = True
        self.current_density_profile = None
        self.HII_radius = None
        self.n0 = None # standard particle density background gas

        # toggle variables, they get toggled on somewhere else
        self.gravity_star_on = False
        self.gravity_clumps_on = False
        self.gravity_BGG_on = False

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
                m = data_clumps[clump_name]['mass'] * m_sun
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
                d_max = boundary_box[1]
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

    def Import_weltgeist_data(self, weltgeist_data_file):
        """
        This function imports all data obtained by the Weltgeist code. This
        is thus far only the density, nH, for each radius and all time periods.
        """
        file = pd.read_excel(r'C:\Users\Hiele\Documents\School\Jaar 5 UvA 2019-20\Scriptie\Thesis\Thesis\measurements\HII_region_Weltgeist_data/' + weltgeist_data_file + '.xlsx')

        self.weltgeist_data = {}
        niterations = int((len(pd.DataFrame(file).values) + 1) / 4)
        for i in range(niterations):
            time = pd.DataFrame(file).values[i * 4][1]
            x = pd.DataFrame(file).values[i * 4 + 1]
            nH = pd.DataFrame(file).values[i * 4 + 2]
            densities = {}
            for i in range(1, len(x) - 1):
                densities[x[i] / 100] = nH[i] * 1e6

            self.weltgeist_data[time] = densities

        # retrieve first HII radius value
        self.Get_nH_profile()
        self.Get_HII_radius()

    def Get_nH_profile(self):
        """
        This function get the current density profile of the background gas.
        """
        # when using dummy data for the particle density
        if self.use_weltgeist_dummy_data:
            HII_radius = 0.65 * pc / Myr * self.time # the HII region has a radius of 6.5pc after 10 Myr
            size_cell = 15 * pc / 512
            self.current_density_profile = {}
            for i in range(512):
                current_radius = size_cell * i
                if current_radius < HII_radius:
                    self.current_density_profile[size_cell * i] = 0
                else:
                    self.current_density_profile[size_cell * i] = self.n0

        # when using the actual weltgeist data
        else:
            remove_keys = []
            for time_weltgeist in self.weltgeist_data:
                if time_weltgeist > self.time:
                    self.current_density_profile = self.weltgeist_data[time_weltgeist]
                    break
                remove_keys.append(time_weltgeist)

            # remove radii with a corresponding time less than the current time
            # this prevents unnecessary looping over a lot of data, decreasing runtime
            for remove_key in remove_keys:
                del self.weltgeist_data[remove_key]

    def Get_HII_radius(self):
        """
        This function returns the finds the radius of the HII region, which is
        the radius with the highest particle density (nH).
        """
        # when using dummy data to safe time. Often to test functions
        if self.use_weltgeist_dummy_data:
            # print("test")
            self.HII_radius = 0.65 * pc / Myr * self.time # the HII region has a radius of 6.5pc after 10 Myr

        # when using the actual Weltgeist data
        else:
            highest_nH = 0
            for x in self.current_density_profile:
                nH = self.current_density_profile[x]
                if nH > highest_nH:
                    highest_nH = nH
                    highest_x = x
            self.HII_radius = highest_x

    def Step(self):
        """
        This function executes one timestep of length dt using Eulers method.
        """
        # reset accelerations of clumps from previous iterations
        # self.star.ax = 0
        # self.star.ay = 0
        # self.star.az = 0
        for clump in self.clumps:
            clump.ax = 0
            clump.ay = 0
            clump.az = 0

        # update new particle density (nH) profile for this time step
        self.Get_nH_profile()

        # determine new CM of cloud
        self.Find_CM()

        # update the new HII region radius
        self.Get_HII_radius()

        # calculate all forces and acceleration
        if self.gravity_star_on:
            self.Gravity_star()
        if self.gravity_clumps_on:
            self.Gravity_clumps()
        if self.gravity_BGG_on:
            self.Gravity_BGG()

        # # calcualte new position of star (TODO, keep everyting relative to the star and have it always at (0,0,0))
        # # This piece of code is used when the position of the star is NOT fixed
        # self.star.vx += self.star.ax * self.dt
        # self.star.vy += self.star.ay * self.dt
        # self.star.vz += self.star.az * self.dt
        #
        # self.star.x += self.star.vx * self.dt
        # self.star.y += self.star.vy * self.dt
        # self.star.z += self.star.vz * self.dt

        # calculate new positions for each clump
        for clump in self.clumps:
            clump.vx += clump.ax * self.dt
            clump.vy += clump.ay * self.dt
            clump.vz += clump.az * self.dt

            clump.x += clump.vx * self.dt
            clump.y += clump.vy * self.dt
            clump.z += clump.vz * self.dt

            # if clumps have escaped the cloud and are beyond view, remove them
            if abs(clump.x) > 3 * self.boundary_box[1]:
                self.clumps.remove(clump)
            elif abs(clump.y) > 3 * self.boundary_box[1]:
                self.clumps.remove(clump)
            elif abs(clump.z) > 3 * self.boundary_box[1]:
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
                           / self.boundary_box[1]**2, c='blue')


            ax.set_zlim(self.boundary_box)
            ax.set_zlabel('Distance (pc)')
            ax.set_zticks(distance_values)
            ax.set_zticklabels(axis_labels)

        # plot of 2D state
        else:
            ax = fig.add_subplot(111)

            # plot star
            if self.star:
                plt.scatter(self.star.x, self.star.y, label="Star",\
                            facecolor="red")

            # plot clumps
            for clump in self.clumps:
                plt.scatter(clump.x, clump.y, s=1.24e6 * clump.R**2 * \
                            self.size_box**(-2), label = "Clump", \
                            facecolor = "blue")

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
        plt.title("State of cloud after %.1f Myr" %(self.time / Myr))
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

    def Mass_BGG(self, R):
        """
        This function calculates the mass of the background gas within a given
        radius
        """
        mass_sum = 0
        R_previous= 0
        for R_current in self.current_density_profile:
            if R_current > R:
                break
            V_shell = 4 / 3 * np.pi * (R_current**3 - R_previous**3)
            m_shell = self.current_density_profile[R_current] * m_H * V_shell
            mass_sum += m_shell
            R_previous = R_current

        return mass_sum

    def Gravity_BGG(self):
        """
        This function calculate the acceleration of all clumps due to the
        gravitaty produced by the background gas of the entire cloud. The
        mass density of the background gas is always spherical symmetric thus
        the gravity of the gas further away than the clump looked is ignored
        due to symmetry.
        """
        for clump in self.clumps:
            dx = clump.x
            dy = clump.y
            dz = clump.z
            dr = np.sqrt(dx**2 + dy**2 + dz**2)
            m_inside = self.Mass_BGG(dr)
            a = G * m_inside / dr**2
            clump.ax += -a * dx / dr
            clump.ay += -a * dy / dr
            clump.az += -a * dz / dr

    def Gravity_star(self):
        """
        This function calculate the acceleration of all clumps due to the
        gravitational pull of the star
        """
        if not self.star:
            raise Exception("Error, can't calculate gravity of star. There is no star")

        for clump in self.clumps:
            dr, dx, dy, dz = self.Distance(clump, self.star)

            merge_factor = 1
            if dr < merge_factor * (self.star.R + clump.R):
                self.star.R = self.star.R * (self.star.m + clump.m) / self.star.m
                self.star.m += clump.m
                self.clumps.remove(clump)
            ## use this code when the star is NOT kept fixed
            # if not self.Collision(self.star, clump, dr):
                # a_star = G * clump.m / dr**2
                # self.star.ax += -a_star * dx / dr
                # self.star.ay += -a_star * dy / dr
                # self.star.az += -a_star * dz / dr

                a_clump = G * self.star.m / dr**2
                clump.ax += a_clump * dx / dr
                clump.ay += a_clump * dy / dr
                clump.az += a_clump * dz / dr

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
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
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
               "distance_to_CM": distance,
               "impact_angle": theta
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
        ## CM_x = (x1 * m1 + x2 + m2) / (m1 + m2) = numerator / denominator
        # numerator_x = 0
        # numerator_y = 0
        # numerator_z = 0
        # denominator = 0
        # if self.star:
        #     numerator_x += self.star.x * self.star.m
        #     numerator_y += self.star.y * self.star.m
        #     numerator_z += self.star.z * self.star.m
        #     denominator += self.star.m
        # for clump in self.clumps:
        #     numerator_x += clump.x * clump.m
        #     numerator_y += clump.y * clump.m
        #     numerator_z += clump.z * clump.m
        #     denominator += clump.m
        # if self.star or self.clumps:
        #     # This CM is the actual CM
        #     self.CM = [numerator_x / denominator, \
        #                numerator_y / denominator, \
        #                numerator_z / denominator]
        #
        # else:
        #     raise Exception("Can't calculate CM when no clumps of star is initialized")

        # for now I'll keep the CM fixed in the origin (same goes for the star)
        self.CM = [0, 0, 0]

    def Radius_clump(self, mass):
        """
        This function calculates the radius of clumps depending on their mass.
        I derived this formula of excisting data of mass/radius ratios. See
        the file "radius_mass_ratio_clumps.pdf" on my github:
        https://github.com/Hielkes7/Thesis

        R = (M**0.426)/55.55
        R in pc and M in solar masses
        """
        radius = pc / 55.55 * (mass/m_sun)**0.426
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
        mass = 12590 * m_sun * (radius/pc)**2.35
        return mass

    def __str__(self):
        if self.toggle_3D:
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

def animation(state, amount_of_frames, niterations, size_box, animate_CM, animate_HII, weltgeist_data_file):
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

    fig, ax_scat = plt.subplots(1, 1)
    fig.set_size_inches(10, 10) # 10 inches wide and long

    ax_scat.grid(True)

    if animate_HII and state.star:
        # put in background color representing background gas
        back_ground_gas = ax_scat.scatter(0, 0, s=1.24e6, \
                          label = "Background gas", facecolor = "lightsalmon")

    # create scatter template
    scat = ax_scat.scatter(x, y, label = "Gas clumps", facecolor = "blue")
    # HII_region = ax_scat.scatter(x, y, label = "HII region", facecolor = "white")

    # create title template
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
        if animate_HII and state.star and state.HII_radius:
            HII_region = ax_scat.scatter(state.star.x, state.star.y, \
                         s=1.24e6 * state.HII_radius**2 * size_box**(-2), \
                         label = "HII region", facecolor = "#ffffff")

        offsets = []
        sizes = []
        title_scat.set_text(u"{} / {} iterations - {} Myr".format(frame*step_size,\
                       niterations, round(state.time / Myr, 1)))

        # animate star
        if state.star:
            scat_star = ax_scat.scatter(state.star.x, state.star.y, label = "Star", facecolor = "red")

        # animate clumps
        for clump in state.clumps:
            offsets.append([clump.x, clump.y])
            sizes.append(1.24e6 * clump.R**2 * size_box**(-2))
        if state.clumps:
            scat.set_offsets(offsets)
            scat.set_sizes(sizes)

        # centre of mass
        if animate_CM:
            scat_CM = ax_scat.scatter(state.CM[0], state.CM[1], label = "Centre of Mass", facecolor = "green")

        print("Time: %.2f Myr" %round(state.time / Myr, 2))
        print()

        # assemble the return list full with the animation parts
        return_list = []
        if animate_HII and state.star and state.HII_radius:
            return_list.append(back_ground_gas)
            return_list.append(HII_region)
        if animate_CM:
            return_list.append(scat_CM)
        if state.star:
            return_list.append(scat_star)
        return_list.append(scat)
        return_list.append(title_scat)

        # each frame has "step_size" iterations done
        for _ in range(step_size):
            state.Step()

        return return_list

    # blit=True makes it run alot faster but the title gets removed
    myAnimation_scat = FuncAnimation(fig, update_scat, \
                       frames = amount_of_frames, \
                       interval = 10, repeat=True,
                       blit=True)
    plt.show()

def save_plots(state, amount_of_frames, niterations, animate_CM, animate_2D_scatter, animate_3D_scatter, save_collision_distance_to_CM_spectrum, save_mass_spectrum, save_number_density, save_impact_velocity, save_impact_angle_hist, save_impact_angle_vs_distance, animate_HII):
    """
    This function animates evolution of the set up. There is an option for live
    animation or the making of GIF
    """
    # animation settings
    step_size = int(niterations / amount_of_frames) # iterations done between each frame

    # looping all steps and frames
    for frame in range(1, amount_of_frames + 1):

        # information feedback to estimate duration of animation
        duration_calculator(state, frame, amount_of_frames)

        # a string with x amount of 0's to add to the file name number
        filling_zeros = "0" * (4 - len(str(frame)))

        # saving frames for the scatter animation
        if animate_2D_scatter:
            file_name = "frame" + filling_zeros + str(frame)
            save_scatter_frame(state, file_name, animate_CM, True, False, animate_HII)

        # saving frames for the scatter animation
        if animate_3D_scatter:
            file_name = "frame" + filling_zeros + str(frame)
            save_scatter_frame(state, file_name, animate_CM, False, True, animate_HII)

        # have 10 times less plots than frames of the animation
        if frame % 10 == 0:
            # saving plots of the distances of the clump collisions to CM
            if save_collision_distance_to_CM_spectrum:
                file_name= "distance_plot" + filling_zeros + str(frame)
                save_collision_distance_plot(state, file_name)

            # saving plots of the mass spectra
            if save_mass_spectrum:
                file_name= "mass_spectrum" + filling_zeros + str(frame)
                save_mass_spectrum_plot(state, file_name)

            # saving plots of number density of the clumps compared to distance to CM
            if save_number_density:
                file_name= "number_density" + filling_zeros + str(frame)
                save_number_density_plot(state, file_name)

            # saving plots of the impact velocities of clump collisions
            if save_impact_velocity:
                file_name= "collision_impact_velocity" + filling_zeros + str(frame)
                save_impact_velocity_plot(state, file_name)

            # saving histograms of the frequency of impact angles of clump collisions
            if save_impact_angle_hist:
                file_name= "collision_impact_angle_hist" + filling_zeros + str(frame)
                save_impact_angle_hist_plot(state, file_name)

            # saving plots of the impact angles vs distance to CM of clump collisions
            if save_impact_angle_vs_distance:
                file_name= "collision_impact_angle" + filling_zeros + str(frame)
                save_impact_angle_vs_distance_plot(state, file_name)

        for _ in range(step_size):
            state.Step()

def save_scatter_frame(state, file_name, animate_CM, animate_2D_scatter, animate_3D_scatter, animate_HII):
    """
    This function saves a frame of the scatter animation.
    """
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


    # if the simulation is in 2D
    if animate_2D_scatter:
        fig = plt.figure()
        fig.set_size_inches(10, 10) # 10 inches wide and long
        ax = fig.add_subplot(111)

        # plot HII region and background gas
        if animate_HII and state.star:
            plt.scatter(0, 0, s=1.24e6, label = "Background gas", \
            facecolor = "lightblue")

            plt.scatter(state.star.x, state.star.y, s=1.24e6 * state.HII_radius**2\
                        * state.size_box**(-2), label = "HII region", \
                        facecolor = "white")

        # plot clumps
        for clump in state.clumps:
            plt.scatter(clump.x, clump.y, s=1.24e6 * clump.R**2 * \
                        state.size_box**(-2), label = "Clump", \
                        facecolor = "blue")

        # plot star
        if state.star:
            plt.scatter(state.star.x, state.star.y, label="Star",\
                        facecolor="red")

        # plot centre of mass
        if animate_CM:
            plt.scatter(state.CM[0], state.CM[1], label = "Centre of Mass", \
                        facecolor = "green")

        # settings that apply for both 2D and 3D
        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('Distance (pc)')

        ax.set_xticks(distance_values)
        ax.set_xticklabels(axis_labels)
        ax.set_yticks(distance_values)
        ax.set_yticklabels(axis_labels)

        ax.set_xlim(state.boundary_box * 1.5)
        ax.set_ylim(state.boundary_box * 1.5)

        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('Distance (pc)')
        plt.title("State of cloud after %.1f Myr" %(state.time / Myr))
        plt.grid()

        fig.savefig(my_path + "measurements/scatter_frames_2D/" + file_name + ".png")
        plt.close(fig)

    # if the simulation is in 3D
    if animate_3D_scatter:
        fig = plt.figure()
        fig.set_size_inches(10, 10) # 10 inches wide and long
        ax = fig.add_subplot(111, projection='3d')

        # plot HII region and background gas
        if animate_HII and state.star:
            ax.scatter(0, 0, s=1.24e6, label = "Background gas", \
            facecolor = "lightblue", alpha=0.5)

            ax.scatter(state.star.x, state.star.y, state.star.z, s=1.24e6 * state.HII_radius**2\
                        * state.size_box**(-2), label = "HII region", \
                        facecolor = "white")

        # plot star
        if state.star:
            ax.scatter(state.star.x, state.star.y, state.star.z, label="Star",\
                        facecolor="red")

        # plot clumps
        for clump in state.clumps:
            ax.scatter(clump.x, clump.y, clump.z, s=1.24e6 * clump.R**2 * \
                        state.size_box**(-2), label = "Clump", \
                        facecolor = "blue")

        # plot centre of mass
        if animate_CM:
            ax.scatter(state.CM[0], state.CM[1], state.CM[2], label = "Centre of Mass", \
                        facecolor = "green")

        # settings that apply for both 2D and 3D
        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('Distance (pc)')
        ax.set_zlabel('Distance (pc)')

        ax.set_xticks(distance_values)
        ax.set_xticklabels(axis_labels)
        ax.set_yticks(distance_values)
        ax.set_yticklabels(axis_labels)
        ax.set_zticks(distance_values)
        ax.set_zticklabels(axis_labels)

        ax.set_xlim(state.boundary_box)
        ax.set_ylim(state.boundary_box)
        ax.set_zlim(state.boundary_box)

        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('Distance (pc)')
        plt.title("State of cloud after %.1f Myr" %(state.time / Myr))
        plt.grid()

        fig.savefig(my_path + "measurements/scatter_frames_3D/" + file_name + ".png")
        plt.close(fig)

def save_collision_distance_plot(state, file_name):
    """
    This function saves a histogram of the distances of clump collisions to CM.
    """
    # make a list with all distances of all clumps collisions
    distances = []
    for collision in state.collision_data:
        distances.append(collision["distance_to_CM"] / pc)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10) # 10 inches wide and long
    ax.hist(distances, rwidth=0.75)
    ax.set_xlabel('Distance to CM (pc)')
    ax.set_ylabel('Frequency')
    plt.title("Distance spectrum of clump collisions after %.1f Myr. Initially %d clumps" %(state.time / Myr, state.amount_clumps))
    fig.savefig(my_path + "measurements/collision_distance_plots/" + file_name + ".png")
    plt.close(fig)

def save_mass_spectrum_plot(state, file_name):
    """
    This function saves a histogram of the mass specturm of the clumps of
    the current state
    """
    # make a list with all clump masses
    clump_masses = []
    for clump in state.clumps:
        clump_masses.append(clump.m / m_sun)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10) # 10 inches wide and long
    ax.hist(clump_masses, rwidth=0.75)
    ax.set_xlabel('Mass ($M_{sun}$)')
    ax.set_ylabel('Frequency')
    plt.title("Mass spectrum of the clumps after %.1f Myr. Initially %d clumps" %(state.time / Myr, state.amount_clumps))
    fig.savefig(my_path + "measurements/mass_spectrum_plots/" + file_name + ".png")
    plt.close(fig)

def save_number_density_plot(state, file_name):
    """
    This function saves a histogram of the mass specturm of the clumps of
    the current state
    """
    # make a list with all clump masses
    distances = []
    for clump in state.clumps:
        distances.append(np.sqrt(clump.x**2 + clump.y**2 + clump.z**2) / pc)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10) # 10 inches wide and long
    ax.hist(distances, rwidth=0.75)
    ax.set_xlabel('Distance to CM (pc)')
    ax.set_ylabel('Frequency')
    plt.title("Number density of the clumps per distance to CM. Time = %.1f Myr. Initially %d clumps" %(state.time / Myr, state.amount_clumps))
    fig.savefig(my_path + "measurements/number_density_plots/" + file_name + ".png")
    plt.close(fig)

# def plot_collisions(state, niterations):
#     """
#     This function runs the whole simulation till the end and then plots data
#     about the clump collisions.
#     """
#     plot_mass_spectrum = False
#     plot_distance_spectrum = True
#
#     for iteration in range(niterations):
#         print(iteration)
#         state.Step()
#
#     # histogram of mass spectrum
#     if plot_mass_spectrum:
#         fig, ax = plt.subplots(1, 1)
#         fig.set_size_inches(10, 10) # 10 inches wide and long
#
#         # make a list with all masses of all clumps
#         all_masses = []
#         for clump in state.clumps:
#             all_masses.append(clump.m / m_sun)
#
#         ax.cla()
#         ax.hist(all_masses, rwidth=0.75)
#         ax.set_xlabel('Mass ($M_{sun}$)')
#         ax.set_ylabel('Frequency')
#         plt.title("Mass spectrum after %.1f Myr" %(state.time / Myr))
#         plt.show()
#
#     if plot_distance_spectrum:
#         fig, ax = plt.subplots(1, 1)
#         fig.set_size_inches(10, 10) # 10 inches wide and long
#
#         # make a list with all masses of all clumps
#         distances = []
#         for collision in state.collision_data:
#             distances.append(collision["distance_to_CM"] / pc)
#
#         ax.cla()
#         ax.hist(distances, rwidth=0.75)
#         ax.set_xlabel('Distance to CM (pc)')
#         ax.set_ylabel('Frequency')
#         plt.title("Distance spectrum of collisions after %.1f Myr" %(state.time / Myr))
#         plt.show()

def save_impact_velocity_plot(state, file_name):
    """
    This function saves a plot of the collision velocity vs the distance to CM.
    """
    # make a list with all distances of all clumps collisions
    distances = []
    velocities = []
    for collision in state.collision_data:
        distances.append(collision["distance_to_CM"] / pc)
        velocities.append(collision["impact_velocity"] * Myr / pc)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10) # 10 inches wide and long
    plt.scatter(distances, velocities)
    ax.set_xlabel('Distance to CM (pc)')
    ax.set_ylabel('Impact velocity (pc/Myr)')
    plt.title("The impact velocity vs distance to CM after %.1f Myr. Initially %d clumps" %(state.time / Myr, state.amount_clumps))
    fig.savefig(my_path + "measurements/impact_velocity_plots/" + file_name + ".png")
    plt.close(fig)

def save_impact_angle_hist_plot(state, file_name):
    """
    This function saves a hist of the impact angles of clump collisions
    """
    # make a list with all distances of all clumps collisions
    angles = []
    for collision in state.collision_data:
        angles.append(collision["impact_angle"] / (2 * np.pi) * 360)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10) # 10 inches wide and long
    ax.hist(angles, rwidth=0.75)
    ax.set_xlabel('Impact angles of clump collisions (degree)')
    ax.set_ylabel('Frequency')
    plt.title("The impact angles of clump collisions after %.1f Myr. Initially %d clumps" %(state.time / Myr, state.amount_clumps))
    fig.savefig(my_path + "measurements/impact_angle_hist/" + file_name + ".png")
    plt.close(fig)

def save_impact_angle_vs_distance_plot(state, file_name):
    """
    This function saves a plot of the impact angle vs the distance of the collision to CM
    """
    # make a list with all distances of all clumps collisions
    angles = []
    distances = []
    for collision in state.collision_data:
        angles.append(collision["impact_angle"] / (2 * np.pi) * 360)
        distances.append(collision["distance_to_CM"] / pc)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10) # 10 inches wide and long
    plt.scatter(distances, angles)
    ax.set_xlabel('Distance to CM (pc)')
    ax.set_ylabel('Impact angles of clump collisions (degree)')
    plt.title("The impact angles vs their distance to CM after %.1f Myr. Initially %d clumps" %(state.time / Myr, state.amount_clumps))
    fig.savefig(my_path + "measurements/impact_angle_vs_distance/" + file_name + ".png")
    plt.close(fig)

def duration_calculator(state, frame, amount_of_frames):
    """
    This function keeps track of how long the program has been iterating
    and how long it will take till it's finished
    """
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
          sec_left)) #, end="\r")
    print("Clumps: %d" %len(state.clumps))
    print()

def set_up():
    """
    This function contains all the input values. The user adjusts them.
    """
    # simulation settings
    time_frame =  5 * Myr # time frame in 10^6 years
    niterations = 1000
    size_box = 13 * pc # diameter of orbit of pluto
    toggle_3D = True

    # animation settings
    boundary_box = -size_box/2, size_box/2
    amount_of_frames = 100
    dt = time_frame / niterations # s
    weltgeist_data_file = "HII region expansion"

    # star settings
    # minimum visable radius is size_box / 1000
    R_star = 40 * R_sun # radius sun displayed in anmiation, not in scale
    M_star = 35 * m_sun
    QH = 1e45 # photon per second emitted

    # clump settings
    use_Sams_data = False
    amount_clumps = 30
    cloud_mass = 3400 * m_sun # obtained from the data Sam gave me, not containing background gas yet
    initial_clump_mass = cloud_mass / amount_clumps
    max_velocity_fraction = 0
    curl_fraction = 0.8
    random_movement_fraction = 1

    # background gas settings
    n0 = 1e8 # m^-3, standard particle density of the background gas_pressure_on
    import_weltgeist_data = False # if this is False, the program will use dummy data
    animate_HII = True

    # choose one
    clump_distribution = "constant"
    # clump_distribution = "power_law"

    make_animation = True
    animate_CM = False
    init_star = True

    # SAVING PLOTS
    save_data = False # if this value is False, all other toggle save data's are ignored
    animate_2D_scatter = True
    animate_3D_scatter = True
    save_collision_distance_to_CM_spectrum = True
    save_mass_spectrum = True
    save_number_density = True
    save_impact_velocity = True
    save_impact_angle_hist = True # this figure just counts the different impact angles
    save_impact_angle_vs_distance = True # this figure compares the different impact angles to their distance to CM


    # initializing begin state
    state = State(toggle_3D, \
                  amount_clumps, \
                  dt, \
                  boundary_box, \
                  clump_distribution, \
                  max_velocity_fraction, \
                  curl_fraction, \
                  cloud_mass, \
                  initial_clump_mass, \
                  use_Sams_data, \
                  random_movement_fraction)
    state.n0 = n0
    if init_star:
        state.Initiate_star(R_star, M_star)
    if import_weltgeist_data:
        state.use_weltgeist_dummy_data = False
        state.Import_weltgeist_data(weltgeist_data_file)


    # toggle force parameters
    state.gravity_star_on = False
    state.gravity_clumps_on = False
    state.gravity_BGG_on = True

    state.clump_evaportation_on = False # TODO
    state.stellar_wind_on = False # TODO
    state.radiation_pressure_on = False # TODO
    state.gas_pressure_on = False # TODO

    if niterations < amount_of_frames:
        raise Exception("Amount_of_frames is higher than niterations")

    if make_animation and save_data:
        raise Exception("Both make_animation and save_data are true. That's not allowed in this program.")

    if make_animation:
        animation(state, \
                  amount_of_frames, \
                  niterations, \
                  size_box, \
                  animate_CM, \
                  animate_HII, \
                  weltgeist_data_file)

    elif save_data:
        save_plots(state, \
                   amount_of_frames, \
                   niterations, \
                   animate_CM, \
                   animate_2D_scatter, \
                   animate_3D_scatter, \
                   save_collision_distance_to_CM_spectrum, \
                   save_mass_spectrum, \
                   save_number_density, \
                   save_impact_velocity, \
                   save_impact_angle_hist, \
                   save_impact_angle_vs_distance, \
                   animate_HII)

if __name__ == "__main__":
    set_up()
