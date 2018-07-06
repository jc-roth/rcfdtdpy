from sim import Sim
from sim import Field
import vis
import numpy as np
from matplotlib import pyplot as plt
"""
A module that shows how to use the sim module to prepare and run RC-FDTD simulations.
"""

def prep_sim():
    """
    Used to prepare the simulation object
    """

def source_calc(n):
    #return np.exp(-((n-20)**2)/10)
    #return np.sin(2*np.pi*(n/2))
    #return (-1)**n
    pass

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
    delta_z = delta_t
    
    dim_n = 100
    dim_i = 100

    # Prepare current field
    cfield = Field(dim_n, dim_i)
    current = np.zeros((dim_n, dim_i))

    x = np.arange(0, dim_n+2, 1)

    current[:, 49] = np.diff(np.diff(np.exp(-((x-20)**2)/(5))))

    cfield = Field(field=current)

    # Prepare and perform simulation
    s = Sim(vacuum_permittivity, infinity_permittivity, vacuum_permability, delta_t, delta_z, dim_n, dim_i, cfield, 0, initial_susceptability)
    s.simulate()

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
    #vis.plot(e, h, cfield.export(), loc)
    vis.timeseries(e, h, cfield.export(), '../temp/plt')