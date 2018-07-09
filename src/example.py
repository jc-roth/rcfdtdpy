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
    c0 = 1 # 300 um/ps (speed of light)

    dn = 0.05 # 0.05 ps
    n0 = 0 # 0 ps
    n1 = 10 # 4 ps

    di = dn * c0 # (300 um/ps)(0.05 ps) = 15 um
    i0 = -di*1000 # -15000 um
    i1 = di*1000 # 15000 um

    vacuum_permittivity = 1
    vacuum_permeability = 1
    infinity_permittivity = 1
    susceptibility = 0
    initial_susceptibility = 0

    # Prepare current field

    cfield = Field(i0, i1, di, n0, n1, dn) # Create a field of the correct dimensions
    cn, ci = cfield.dims() # Determine the dimensions
    c = cfield.export() # Export the field into a useable format
    del cfield # Delete the old field

    t = np.linspace(n0, n1, cn, False) # Create a time stream
    tc = 5 # Center the pulse at 0.5ps
    c[:, 999] = np.append(np.diff(np.diff(np.exp(-((t-tc)**2)))), [0,0]) # Generate a Gaussian pulse

    cfield = Field(i0, i1, di, n0, n1, dn, field=c) # Create a new field from the Numpy 2D array

    #plt.plot(cfield.export()[:,999])
    #plt.show()



    # Prepare and perform simulation
    s = Sim(i0, i1, di, n0, n1, dn, vacuum_permittivity, infinity_permittivity, vacuum_permeability, susceptibility, initial_susceptibility, cfield)
    s.simulate()

    # Export simulation result
    e = s.get_efield().export()
    h = s.get_hfield().export()

    #vis.contor_plot(e, h)
    #vis.plot(e, h, cfield.export(), loc)
    vis.timeseries(e, h, cfield.export(), '../temp/plt')