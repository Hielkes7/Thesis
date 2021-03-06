# all data about the figures Sam gave me

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

masses_log = [0.08, 0.39, 0.88, 1.08, 1.12, 1.21, 1.48, \
              1.60, 1.66, 1.85, 1.89, 1.92, 2.00, 2.15, \
              2.21, 2.28, 2.34, 2.65, 3.24]

masses = [1.202264434617413, 2.4547089156850306, 7.5857757502918375, \
          12.02264434617413, 13.182567385564074, 16.218100973589298, \
          30.19951720402016, 39.810717055349734, 45.708818961487495, \
          70.79457843841381, 77.62471166286916, 83.17637711026708, \
          100.0, 141.2537544622754, 162.18100973589299, 190.54607179632464, \
          218.77616239495518, 446.683592150963, 1737.8008287493763]

radii_log = [-1.43, -1.43, -1.43, -1.34, -1.23, -1.23, -1.23, \
              -1.19, -1.13, -1.08, -1.06, -1.01, -0.86, -0.82, \
              -0.78, -0.74, -0.65, -0.65, -0.43]

radii = [0.03715352290971726, 0.03715352290971726, 0.03715352290971726, \
         0.04570881896148749, 0.0588843655355589, 0.0588843655355589, \
         0.0588843655355589, 0.06456542290346556, 0.07413102413009177, \
         0.08317637711026708, 0.08709635899560805, 0.09772372209558107, \
         0.1380384264602885, 0.15135612484362082, 0.16595869074375605, \
         0.18197008586099836, 0.22387211385683395, 0.22387211385683395, \
         0.37153522909717257]

distances_log = [0.75, 0.49, 0.74, 0.48, 0.46, 0.61, 0.39, \
                0.76, 0.40, 0.52, 0.33, 0.20, 0.35, 0.17, \
                0.47, 0.18, 0.29, 0.25, 0.25]

distances = [5.623413251903491, 3.0902954325135905, 5.495408738576246, \
             3.019951720402016, 2.884031503126606, 4.073802778041127, \
             2.4547089156850306, 5.7543993733715695, 2.51188643150958, \
             3.311311214825911, 2.137962089502232, 1.5848931924611136, \
             2.2387211385683394, 1.4791083881682074, 2.9512092266663856, \
             1.5135612484362082, 1.9498445997580451, 1.7782794100389228, \
             1.7782794100389228]

total_mass = 3397.222201528117

# data_clumps_log = {}
# for i in range(1,20):
#     name = "clump" + str(i)
#     data_clumps_log[name] = {"mass_log": masses_log[i-1],
#                              "mass": masses[i-1],
#                              "radius_log": radii_log[i-1],
#                              "radius": radii[i-1],
#                              "distance_log": distances_log[i-1],
#                              "distance": distances[i-1]}

data_clumps_log =  {'clump1': {'mass_log': 0.08,
                               'mass': 1.202264434617413,
                               'radius_log': -1.43,
                               'radius': 0.03715352290971726,
                               'distance_log': 0.75,
                               'distance': 5.623413251903491},
                    'clump2': {'mass_log': 0.39,
                               'mass': 2.4547089156850306,
                               'radius_log': -1.43,
                               'radius': 0.03715352290971726,
                               'distance_log': 0.49,
                               'distance': 3.0902954325135905},
                    'clump3': {'mass_log': 0.88,
                               'mass': 7.5857757502918375,
                               'radius_log': -1.43,
                               'radius': 0.03715352290971726,
                               'distance_log': 0.74,
                               'distance': 5.495408738576246},
                    'clump4': {'mass_log': 1.08,
                               'mass': 12.02264434617413,
                               'radius_log': -1.34,
                               'radius': 0.04570881896148749,
                               'distance_log': 0.48,
                               'distance': 3.019951720402016},
                    'clump5': {'mass_log': 1.12,
                               'mass': 13.182567385564074,
                               'radius_log': -1.23,
                               'radius': 0.0588843655355589,
                               'distance_log': 0.46,
                               'distance': 2.884031503126606},
                    'clump6': {'mass_log': 1.21,
                               'mass': 16.218100973589298,
                               'radius_log': -1.23,
                               'radius': 0.0588843655355589,
                               'distance_log': 0.61,
                               'distance': 4.073802778041127},
                    'clump7': {'mass_log': 1.48,
                               'mass': 30.19951720402016,
                               'radius_log': -1.23,
                               'radius': 0.0588843655355589,
                               'distance_log': 0.39,
                               'distance': 2.4547089156850306},
                    'clump8': {'mass_log': 1.6,
                               'mass': 39.810717055349734,
                               'radius_log': -1.19,
                               'radius': 0.06456542290346556,
                               'distance_log': 0.76,
                               'distance': 5.7543993733715695},
                    'clump9': {'mass_log': 1.66,
                               'mass': 45.708818961487495,
                               'radius_log': -1.13,
                               'radius': 0.07413102413009177,
                               'distance_log': 0.4,
                               'distance': 2.51188643150958},
                    'clump10': {'mass_log': 1.85,
                               'mass': 70.79457843841381,
                               'radius_log': -1.08,
                               'radius': 0.08317637711026708,
                               'distance_log': 0.52,
                               'distance': 3.311311214825911},
                    'clump11': {'mass_log': 1.89,
                               'mass': 77.62471166286916,
                               'radius_log': -1.06,
                               'radius': 0.08709635899560805,
                               'distance_log': 0.33,
                               'distance': 2.137962089502232},
                    'clump12': {'mass_log': 1.92,
                               'mass': 83.17637711026708,
                               'radius_log': -1.01,
                               'radius': 0.09772372209558107,
                               'distance_log': 0.2,
                               'distance': 1.5848931924611136},
                    'clump13': {'mass_log': 2.0,
                               'mass': 100.0, 'radius_log': -0.86,
                               'radius': 0.1380384264602885,
                               'distance_log': 0.35,
                               'distance': 2.2387211385683394},
                    'clump14': {'mass_log': 2.15,
                               'mass': 141.2537544622754,
                               'radius_log': -0.82,
                               'radius': 0.15135612484362082,
                               'distance_log': 0.17,
                               'distance': 1.4791083881682074},
                    'clump15': {'mass_log': 2.21,
                               'mass': 162.18100973589299,
                               'radius_log': -0.78,
                               'radius': 0.16595869074375605,
                               'distance_log': 0.47,
                               'distance': 2.9512092266663856},
                    'clump16': {'mass_log': 2.28,
                               'mass': 190.54607179632464,
                               'radius_log': -0.74,
                               'radius': 0.18197008586099836,
                               'distance_log': 0.18,
                               'distance': 1.5135612484362082},
                    'clump17': {'mass_log': 2.34,
                               'mass': 218.77616239495518,
                               'radius_log': -0.65,
                               'radius': 0.22387211385683395,
                               'distance_log': 0.29,
                               'distance': 1.9498445997580451},
                    'clump18': {'mass_log': 2.65,
                               'mass': 446.683592150963,
                               'radius_log': -0.65,
                               'radius': 0.22387211385683395,
                               'distance_log': 0.25,
                               'distance': 1.7782794100389228},
                    'clump19': {'mass_log': 3.24,
                               'mass': 1737.8008287493763,
                               'radius_log': -0.43,
                               'radius': 0.37153522909717257,
                               'distance_log': 0.25,
                               'distance': 1.7782794100389228}}

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

clump = data_clumps_log["clump1"]
print("Mass:      ", round(clump["mass"]), "M_sun / ", clump["mass"]*mass_sun, "kg")
print("Radius:    ", round(clump["radius"],3), "pc / ", clump["radius"]*pc, "m")
print("Dist:      ", round(clump["distance"], 3), "pc / ", clump["distance"]*pc, "m")
print("Mass_log:  ", clump["mass_log"])
print("Radius_log:", clump["radius_log"])
print("Dist_log:  ", clump["distance_log"])

print("Mass calc:", Mass_clump(7.7e15))
print("Radius calc:", Radius_clump(34*mass_sun)/pc, "pc")























#
