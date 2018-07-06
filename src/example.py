from sim import Sim
from sim import Field
import vis
import numpy as np
"""
A module that shows how to use the sim module to prepare and run RC-FDTD simulations.
"""

def prep_sim():
    """
    Used to prepare the simulation object
    """

def source_calc(n):
    return np.exp(-((n-1)**2)/2)
    #return np.sin(2*np.pi*(n/2))
    #return (-1)**n

if __name__ == '__main__':
    
    # Prepare constants
    vacuum_permittivity = 8.854187817e-12 # F/m
    vacuum_permability = 1.2566370614e-6 # N/A^2
    infinity_permittivity = 1
    initial_susceptability = 0
    delta_t = 1e-12 # s (i.e. one picosecond)
    delta_z = delta_t * (3e8)

    # Prepare constants
    vacuum_permittivity = 1
    vacuum_permability = 1
    infinity_permittivity = 1
    initial_susceptability = 0
    delta_t = 1
    delta_z = delta_t * 0.004

    loc = 10

    dim_n = loc + 1
    dim_i = 500

    # Prepare current field
    cfield = Field(dim_n, dim_i)
    for n in range(dim_n):
        cfield.set_time_index(n)
        cfield[250] = source_calc(n)

    # Prepare and perform simulation
    s = Sim(vacuum_permittivity, infinity_permittivity, vacuum_permability, delta_t, delta_z, dim_n, dim_i, cfield, 0, initial_susceptability)
    s.simulate(True, (245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255))

    # Export simulation result
    e = s.get_efield().export()
    h = s.get_hfield().export()
    #arr = np.load('sim.npy')

    # Save the result
    #np.save('sim.npy', arr)

    # Visualize result
    print(np.shape(e))
    print(np.shape(h))
    #vis.contor_plot(e, h)
    #vis.plot(e, h, loc)