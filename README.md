RC-FDTD
=======

Ideas
----
* For a slight speedup, consider making the `_get_current()` function dynamic, so that the current does not need to be calculated twice, once for the reference E-field and once for the normal E-field
* Replace the Numpy arrays with h5py datasets, which can store much larger files locally.

Todo
----
* Update docs so that Mat and Sim have separate pages on the documentation
* Add a tips section
    * One tip: if the field lines look very thick, move forward your current pulse in time. It is also helpful for current pulses to be smooth (i.e. no step or delta functions), as these discontinuous functions seem to mess up simulation math
* Create a simulation with higher spatial resolution and also increase the simulation time. Increase the width of the current pulse, meaning a higher spread of frequencies will go into the pulse meaning the coefficients of a larger region can be explored.
* Check results against Ben's results
* Ask Ben about his calculation of chi_0_2 (on line 81 of his original code). Isn't he missing a *-1?