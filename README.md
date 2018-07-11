RC-FDTD
=======

Todo
----
* Change the boundaries of the simulation so that they *mirror* the cells inside the simulation (or perhaps inverse mirror, not sure which will work yet). This will prevent reflection at the boundaries, and enable waves to propegate out of the simulation.
* Ensure that units are plotting correctly in the vis module
* Test the mirror and periodic boundary conditions
* Add unit label to the pcolormesh levels
* Create a 'builder' class which can be used to easily create current and susceptability matrices/tensors
* Can the loops in :code:`_calc_dchi` be removed, with all calculations being done entirely on tensors?
* Create reference fields like Ben has