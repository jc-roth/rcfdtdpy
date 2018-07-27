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
* Remake Mat object so that the entire material contains the same properties. This will allow a single psi value to be calculated for the material, which will perhaps stop fields in the material blowing up after a sufficiently long time.