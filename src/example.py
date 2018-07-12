from sim import Sim
import vis
import numpy as np
from matplotlib import pyplot as plt
"""
A module that shows how to use the sim module to prepare and run RC-FDTD simulations.
"""

if __name__ == '__main__':

    """ Ben's constants
    # Prepare constants
    c0 = 1 # 300 um/ps (speed of light)

    di = 0.004 # 0.004 um
    i0 = di * -250 # 0.004 um * -250 = -1 um
    i1 = di * 500 # 0.004 um * 500 = 2 um

    dn = di / c0 # (0.004 um) / (300 um/ps)
    n0 = 0/dn
    n1 = 20/dn
    """

    # Prepare constants
    c0 = 1 # 300 um/ps (speed of light)

    di = 0.05 # 0.05 um
    i0 = di * -200 # 0.05 um * -200 = -0.1 um
    i1 = di * 200 # 0.05 um * 200 = 0.1 um

    dn = di / c0 # (0.005 um) / (300 um/ps)
    n0 = 0*dn
    n1 = 750*dn

    vacuum_permittivity = 1
    vacuum_permeability = 1
    infinity_permittivity = 1
    initial_susceptibility = 0

    # Prepare current field
    nlen, ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
    c = np.zeros((nlen, ilen))
    t = np.multiply(np.arange(0, nlen, 1), dn) # Create a time stream
    t_center = t[int(nlen/10)]
    loc_center = int(ilen/4)
    c[:, loc_center] = np.append(np.diff(np.diff(np.exp(-((t-t_center)**2)))), [0,0]) # Generate a Gaussian pulse

    # Plot current in time before proceeding
    #plt.plot(c[:,loc_center])
    #plt.show()

    # Prepare material
    a1 = np.complex64(2.5)
    a2 = np.complex64(0)
    gamma = np.complex64(1)
    omega = np.complex64(0.005)

    beta = np.sqrt(np.add(np.square(gamma), -np.square(omega)))
    mat = np.ones((1, 200), dtype=np.complex64)
    mata1 = mat * a1
    mata2 = mat * a2
    matg = mat * gamma
    matb = mat * beta

    # Create and start simulation
    s = Sim(i0, i1, di, n0, n1, dn, vacuum_permittivity, infinity_permittivity, vacuum_permeability, c, 'absorbing', 202, mata1, mata2, matg, matb, nstore=300, storelocs=[1,ilen-1])
    s.simulate()
    # Visualize
    vis.timeseries(s, iscale=1, interval=20, iunit='$\mu$m', eunit='N/c', hunit='A/m')
    #vis.plot_loc(s)