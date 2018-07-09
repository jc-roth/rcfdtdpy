RC-FDTD
=======

Todo
----
* Convert the sim.Sim class so that instead of the user defining \Delta z and \Delta t and then the number of \Delta t and \Delta z to have, the user simply defines \Delta z and \Delta t and then the field boundaries (i.e. 10 microns wide, or something similar)
* Change the `_calc_hfield` and `_calc_efield` functions in the `sim.Sim` class so that they compute their values using array operations, not by looping through the array indicies. This will make for more efficient calculations (it takes advantage of Numpy's built-in optimizations).
* Change the boundaries of the simulation so that they *mirror* the cells inside the simulation (or perhaps inverse mirror, not sure which will work yet). This will prevent reflection at the boundaries, and enable waves to propegate out of the simulation.