RC-FDTD
=======

Ideas
----
* For a slight speedup, consider making the `_get_current()` function dynamic, so that the current does not need to be calculated twice, once for the reference E-field and once for the normal E-field
* Replace the Numpy arrays with h5py datasets, which can store much larger files locally.

Todo
----
* RC-FDTD v2
    * Create a material whose properties are determined via numerical integration, implement another material class that stores past values of the electric field within its boundaries in order to do so.
* Add a tips section
    * One tip: if the field lines look very thick, move forward your current pulse in time. It is also helpful for current pulses to be smooth (i.e. no step or delta functions), as these discontinuous functions seem to mess up simulation math