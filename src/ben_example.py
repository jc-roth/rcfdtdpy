from sim import Sim, Mat
import vis
import numpy as np
from matplotlib import pyplot as plt
"""
A module that shows how to use the sim module to prepare and run RC-FDTD simulations.
"""

if __name__ == '__main__':

    # Prepare constants
    c0 = 1 # 300 um/ps (speed of light)

    di = 0.004 # 0.004 um
    i0 = -1 # 0.004 um * -250 = -1 um
    i1 = 2 # 0.004 um * 500 = 2 um

    dn = di / c0 # (0.004 um) / (300 um/ps)
    n0 = -4.0
    n1 = 10.0

    vacuum_permittivity = 1
    vacuum_permeability = 1
    mat_infinity_permittivity = 16

    # Prepare current field
    nlen, ilen = Sim.calc_dims(n0, n1, dn, i0, i1, di)
    c = np.zeros((nlen, ilen))
    z = np.arange(i0, i1, di) # Create a location stream
    t = np.arange(n0, n1, dn) # Create a time stream
    
    t_center = 0.0
    z_center = 0
    pulse = np.append(np.diff(np.diff(np.exp(-((t-t_center)**2)/(2*0.05)))), [0,0])

    z_center_ind = np.argmin(np.abs(np.subtract(z, z_center)))
    c[:,z_center_ind] = 100*pulse/(np.max(np.abs(pulse)))

    # Prepare material values
    mat0 = 1.99
    mat1 = 2.00
    gamma = np.complex64(0.01 * 2 * np.pi)
    omega = np.complex64(0.0)
    beta = np.sqrt(np.add(np.square(gamma), -np.square(omega)))
    a1 = np.complex64(160/(2*beta))
    a2 = np.complex64(-160/(2*beta))

    # Create material matrix
    mstart, mlen = Sim.calc_mat_dims(i0, i1, di, mat0, mat1)
    mstart += 1
    mat = np.ones((1, mlen), dtype=np.complex64)
    mata1 = mat * a1
    mata2 = mat * a2
    matg = mat * gamma
    matb = mat * beta

    m = Mat(dn, ilen, mstart, mat_infinity_permittivity, mata1, mata2, matg, matb)
    del m
    m = []

    # Create and start simulation
    s = Sim(i0, i1, di, n0, n1, dn, vacuum_permittivity, vacuum_permeability, c, 'absorbing', m, nstore=int(nlen/4), storelocs=[1,ilen-1])
    s.simulate()
    # Visualize
    vis.timeseries(s, iscale=1, interval=10, iunit=r'$\mu$m', eunit='?', hunit='?')
    #vis.timeseries(s, iscale=1, interval=10, iunit=r'$\mu$m', eunit='?', hunit='?', fname='../temp/sim1.mp4')
    #vis.plot_loc(s, nunit='ps', fname='../temp/sim1.png')