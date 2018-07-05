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

if __name__ == '__main__':
    # Prepare constants
    vacuum_permittivity = 8.854187817e-12
    vacuum_permability = (4 * np.pi)*1e-7
    infinity_permittivity = 1
    initial_susceptability = 0
    delta_z = 3e-6
    delta_t = 1

    dim_n = 10
    dim_i = 500

    # Prepare current field
    cfield = Field(dim_n, dim_i)
    cfield.set_time_index(2)
    cfield[250] = 1

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