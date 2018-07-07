from sim import Sim
from sim import Field
import vis
import numpy as np
from matplotlib import pyplot as plt
"""
A module that shows how to use the sim module to prepare and run RC-FDTD simulations.
"""

if __name__ == '__main__':

    # Prepare constants
    vacuum_permittivity = 1
    vacuum_permability = 1
    infinity_permittivity = 1
    initial_susceptability = 0
    delta_t = 1
    delta_z = delta_t

    n_dim = 250
    i_dim = 500

    # Prepare current field
    ci_index = 250
    cn_index = 10
    cfield = np.zeros((n_dim, i_dim))
    t = np.arange(0, n_dim, 1)
    cfield[:, ci_index] = np.append(np.diff(np.diff(np.exp(-((t-cn_index)**2)/(5)))), [0,0])
    cfield = Field(field=cfield)

    # Prepare and perform simulation
    s = Sim(vacuum_permittivity, infinity_permittivity, vacuum_permability, delta_z, delta_t, n_dim, i_dim, cfield, 0, initial_susceptability)
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