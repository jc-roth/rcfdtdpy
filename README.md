RC-FDTD
=======

Todo
----
* Ensure that units are plotting correctly in the vis module
* Add unit label to the pcolormesh levels
* Create a 'builder' class which can be used to easily create current and susceptability matrices/tensors
* Make the current more like the Mat class, don't require some huge array
* Update docs so that Mat and Sim have separate pages on the documentation
* Add a tips section
    * One tip: if the field lines look very thick, move forward your current pulse in time. It is also helpful for current pulses to be smooth (i.e. no step or delta functions), as these discontinuous functions seem to mess up simulation math
* Work on building the simulation specified by Ben on Friday (see photos app for explanation)

* For a slight speedup, consider making the `_get_current()` function dynamic, so that the current does not need to be calculated twice, once for the reference E-field and once for the normal E-field