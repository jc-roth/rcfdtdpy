from sim import Sim
import vis
import numpy as np
from matplotlib import pyplot as plt
"""
A module that shows how to use the sim module to prepare and run RC-FDTD simulations.
"""

if __name__ == '__main__':

    # Prepare constants
    c0 = 1 # 300 um/ps (speed of light)

    dn = 0.05 # 0.05 ps
    n0 = 0 # 0 ps
    n1 = 1.5 # 15 ps

    di = dn * c0 # (300 um/ps)(0.05 ps) = 15 um
    i0 = di * -200 # 15 um * -100
    i1 = di * 200 # 15 um * 100

    vacuum_permittivity = 1
    vacuum_permeability = 1
    infinity_permittivity = 1
    susceptibility = 0
    initial_susceptibility = 0

    # Prepare current field
    nlen, ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
    c = np.zeros((nlen, ilen))
    t = np.multiply(np.arange(0, nlen, 1), dn) # Create a time stream
    t_center = 4 # Center the pulse at 4ps
    loc_center = 99
    c[:, loc_center] = np.append(np.diff(np.diff(np.exp(-((t-t_center)**2)))), [0,0]) # Generate a Gaussian pulse

    # Plot current in time before proceeding
    #plt.plot(cfield.export()[:,loc_center])
    #plt.show()

    # Create and start simulation
    s = Sim(i0, i1, di, n0, n1, dn, c, 'periodic', vacuum_permittivity, infinity_permittivity, vacuum_permeability, susceptibility, initial_susceptibility)
    print(s)
    s.simulate()

    # Visualize
    vis.timeseries(s, iscale=300, iunit='$\mu$m', eunit='N/c', hunit='A/m')